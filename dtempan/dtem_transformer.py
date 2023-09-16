import torch, torch.nn.functional as F, math, einops, mmengine, numpy, cv2
from torch import nn
from .spatial_loss import Spatial_Loss
from .loss import FocalFrequencyLoss as FFL
from .netutils import ModuleWithInit, BaseModel
from .basic_blocks import ResBlock, conv3x3
from INADToolBox.inadmodel import INADModel


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.L1 = torch.nn.L1Loss()
        self.MSE = torch.nn.MSELoss()
        self.Spatial = Spatial_Loss(in_channels=4)
        self.FFL = FFL(loss_weight=1.0, alpha=1.0)

    def forward(self, output, target):
        loss =  self.MSE(output, target)
        # loss = 10 * self.L1(output, target) + 5 * self.MSE(output, target)
        # loss = self.L1(output, target)
        return loss


@mmengine.MODELS.register_module()
class SFE(ModuleWithInit):
    def __init__(self, in_feats, num_res_blocks, n_feats):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.proj = nn.Sequential(nn.Conv2d(in_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())
        self.RBs = nn.ModuleList()
        for _ in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats))
        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = self.proj(x)
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


@mmengine.MODELS.register_module()
class LFE(nn.Module):
    def __init__(self, in_channels):
        super(LFE, self).__init__()
        self.in_channels = in_channels

        self.conv_64_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn_64_1 = nn.BatchNorm2d(64)
        self.conv_64_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_64_2 = nn.BatchNorm2d(64)

        self.conv_128_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_1 = nn.BatchNorm2d(128)
        self.conv_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_2 = nn.BatchNorm2d(128)

        self.conv_256_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn_256_1 = nn.BatchNorm2d(256)
        self.conv_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn_256_2 = nn.BatchNorm2d(256)

        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.0)

    def forward(self, x):
        out1 = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out1 = self.bn_64_2(self.conv_64_2(out1))

        out1_mp = self.MaxPool2x2(self.LeakyReLU(out1))
        out2 = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out1_mp)))
        out2 = self.bn_128_2(self.conv_128_2(out2))

        out2_mp = self.MaxPool2x2(self.LeakyReLU(out2))
        out3 = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out2_mp)))
        out3 = self.bn_256_2(self.conv_256_2(out3))

        return [out1, out2, out3]


def linestretch(images, tol=None):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().squeeze().numpy()
        images = einops.rearrange(images, "C H W -> H W C")
    if tol is None:
        tol = [0.01, 0.995]
    if images.ndim == 3:
        h, w, channels = images.shape
    else:
        images = numpy.expand_dims(images, axis=-1)
        h, w, channels = images.shape
    images = images.astype(numpy.float16)
    N = h * w
    for c in range(channels):
        image = numpy.float32(numpy.round(images[:, :, c])).reshape(N, 1)
        image = image - numpy.min(image)
        hb, levelb = numpy.histogram(image, bins=math.ceil(image.max() - image.min()))
        chb = numpy.cumsum(hb, 0)
        levelb_center = levelb[:-1] + (levelb[1] - levelb[0]) / 2
        lbc_min, lbc_max = levelb_center[chb > N * tol[0]][0], levelb_center[chb < N * tol[1]][-1]
        image = numpy.clip(image, a_min=lbc_min, a_max=lbc_max)
        image = (image - lbc_min) / (lbc_max - lbc_min)
        images[..., c] = numpy.reshape(image, (h, w))
    images = numpy.squeeze(images)
    return images

@mmengine.MODELS.register_module()
class DTEMTransformer(INADModel, BaseModel, ModuleWithInit):
    def __init__(self, components: dict, bit_depth=11, factor=4, init_cfg=None):
        assert False not in [n in components.keys() for n in ["encoder_lrms_u", "encoder_lrpan_u", "encoder_hrpan", "backbone_lrms", "pem_decoder", "mstm_decoder", "fusion"]], "Missing component."
        self.pem_decoder = None
        self.mstm_decoder = None
        super().__init__(components, init_cfg=init_cfg)
        self.bit_max = float(math.pow(2, bit_depth) - 1)
        self.factor = factor
        self.creterion_res = Criterion()

    def reinit_mstm(self):
        self.mstm_decoder.init_weights()

    def forward(self, x_hrpan, x_lrms, up_lrms=None, gt_hrms=None, mode="predict"):
        if mode == "loss":
            assert gt_hrms is not None
        with torch.no_grad():
            if up_lrms is None:
                up_lrms = F.interpolate(x_lrms, scale_factor=(self.factor, self.factor), mode="bicubic", align_corners=False)
            updown_hrpan = F.interpolate(x_hrpan, scale_factor=(1 / self.factor, 1 / self.factor), mode="bicubic", align_corners=False, recompute_scale_factor=True)
            updown_hrpan = F.interpolate(updown_hrpan, scale_factor=(self.factor, self.factor), mode="bicubic", align_corners=False, recompute_scale_factor=True)
        pf_udpan: list[torch.Tensor] = self.encoder_lrpan_u(updown_hrpan)
        pf_uplrms: list[torch.Tensor] = self.encoder_lrms_u(up_lrms)
        pf_hrpan: list[torch.Tensor] = self.encoder_hrpan(x_hrpan)

        x: list[torch.Tensor] = self.backbone_lrms(x_lrms)
        texture = self.mstm_decoder(x, pf_uplrms, pf_udpan, pf_hrpan) if self.mstm_decoder is not None else None
        edge = self.pem_decoder(x_hrpan, pf_hrpan) if self.pem_decoder is not None else None
        x, res = self.fusion(up_lrms, texture, edge, 1)

        if mode == "predict":
            return x.clamp(0, self.bit_max)
        elif mode == "loss":
            return {"loss": self.creterion_res(res, up_lrms - gt_hrms)}
