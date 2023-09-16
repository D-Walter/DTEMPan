import torch, torch.nn as nn, torch.nn.functional as F, einops, mmengine
from .netutils import ModuleWithInit


class ParC_operator(ModuleWithInit):
    def __init__(self, dim, type, global_kernel_size, use_pe=True):
        super().__init__()
        self.type = type  # H or W
        self.dim = dim
        self.use_pe = use_pe
        self.global_kernel_size = global_kernel_size
        self.kernel_size = (global_kernel_size, 1) if self.type == "H" else (1, global_kernel_size)
        self.gcc_conv = nn.Conv2d(dim, dim, kernel_size=self.kernel_size, groups=dim)
        if use_pe:
            if self.type == "H":
                self.pe = nn.Parameter(torch.randn(1, dim, self.global_kernel_size, 1))
            elif self.type == "W":
                self.pe = nn.Parameter(torch.randn(1, dim, 1, self.global_kernel_size))
            nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        if self.use_pe:
            x = x + self.pe.expand(1, self.dim, self.global_kernel_size, self.global_kernel_size)
        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type == "H" else torch.cat((x, x[:, :, :, :-1]), dim=3)
        x = self.gcc_conv(x_cat)
        return x


class ParC_ConvNext_Block(ModuleWithInit):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, global_kernel_size=14, use_pe=True):
        super().__init__()
        self.gcc_H = ParC_operator(dim // 2, "H", global_kernel_size, use_pe)
        self.gcc_W = ParC_operator(dim // 2, "W", global_kernel_size, use_pe)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x_H, x_W = torch.chunk(x, 2, dim=1)
        x_H, x_W = self.gcc_H(x_H), self.gcc_W(x_W)
        x = torch.cat((x_H, x_W), dim=1)
        x = einops.rearrange(x, "B C H W -> B H W C")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = einops.rearrange(x, "B H W C -> B C H W")

        x = input + self.drop_path(x)
        return x


class ConvNext_Block(ModuleWithInit):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = einops.rearrange(x, "B C H W -> B H W C")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = einops.rearrange(x, "B H W C -> B C H W")

        x = input + self.drop_path(x)
        return x
    
@mmengine.MODELS.register_module()
class ParC_ConvNeXt(ModuleWithInit):
    def __init__(
        self,
        in_chans=4,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        ParC_insert_locs=[3, 3, 6, 2],
        stages_rs=[56, 28, 14, 7],  # gcc block start indices, input resolutions of four stages
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1), LayerNorm(dims[0], eps=1e-6, data_format="channels_first")))
        for i in range(len(depths) - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i, [depth, dim, ParC_insert_loc, input_size] in enumerate(zip(depths, dims, ParC_insert_locs, stages_rs)):
            blocks = []
            for j in range(depth):
                if j < ParC_insert_loc:
                    blocks.append(ConvNext_Block(dim=dim, drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value))
                else:
                    blocks.append(ParC_ConvNext_Block(dim=dim, drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value, global_kernel_size=input_size, use_pe=True))
                stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            cur += depths[i]

    def init_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        pf = []
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            pf.append(x)
        return pf

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def get_model_size(self):
        return sum([p.numel() for p in self.parameters()])


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def parc_convnext_xt(pretrained=False, in_22k=False, **kwargs):
    """
    Constructs a mixed model, where 7*7 depth wise conv operations in the last 1/3 blocks of the last two stages are
    replaced with position aware global circular conv (GCC) operations.
    Args:
        ParC_insert_locs [s1, s2, s3, s4]: ParC_insert_locs[i] indicates that blocks from gcc_bs_indices[i] to end are
        replaced in stage i.
    """

    model = ParC_ConvNeXt(depths=[3, 3, 9, 3], dims=[48, 96, 192, 384], ParC_insert_locs=[3, 3, 6, 2], stages_rs=[56, 28, 14, 7], **kwargs)
    if pretrained:
        raise NotImplementedError("no pretrained model")
    # test
    input = torch.randn(2, 3, 224, 224)
    out = model(input)
    return model
