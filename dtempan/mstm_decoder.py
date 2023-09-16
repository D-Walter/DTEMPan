import torch, torch.nn as nn, torch.nn.functional as F, mmengine, methods.models.dtempan.netutils as netutils
from mmengine.model import BaseModel
from .basic_blocks import ResBlock, conv3x3, LightResBlock
from INADToolBox.attentions import MultiHeadAttention, HeadAttention


@mmengine.MODELS.register_module()
class MultiscaleTextureMaintainingDecoder(BaseModel):
    def __init__(self, ms_ch, ms_dims, pan_dims, num_res_blocks=[1, 1, 1], init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.stage_num = len(num_res_blocks)
        for stage in range(self.stage_num):
            feat_dim = ms_dims[::-1][stage]
            self.__setattr__(f"mha{stage}", MultiHeadAttention(6, ms_dims[self.stage_num - stage - 1], pan_dims[self.stage_num - stage - 1], pan_dims[self.stage_num - stage - 1]))

            rb = [conv3x3(feat_dim, feat_dim)]
            for _ in range(num_res_blocks[stage]):
                rb.append(ResBlock(in_channels=feat_dim, out_channels=feat_dim, activate=nn.CELU))
            self.__setattr__(f"RB1{stage}", nn.Sequential(*rb))

            if stage <= self.stage_num - 2:
                self.__setattr__(f"upsample{stage}", nn.ConvTranspose2d(feat_dim, feat_dim, 2, stride=2, bias=True))
                self.__setattr__(f"chdown{stage}", LightResBlock(in_channels=feat_dim, out_channels=ms_dims[::-1][stage + 1], activate=nn.GELU))

        self.final_proj = nn.Conv2d(in_channels=ms_dims[0], out_channels=ms_dims[0] // 2, kernel_size=3, padding=1)
        self.conv_bypass = nn.Sequential(nn.Conv2d(in_channels=ms_dims[0] // 2, out_channels=ms_ch, kernel_size=3, padding=1))
        self.final_conv = []
        self.RBF = nn.ModuleList()
        for _ in range(4):
            self.RBF.append(ResBlock(in_channels=ms_dims[0] // 2, out_channels=ms_dims[0] // 2))
        self.convF_tail = nn.Sequential(
            nn.Conv2d(in_channels=ms_dims[0] // 2, out_channels=ms_ch, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=ms_ch, out_channels=ms_ch, kernel_size=3, padding=1),
        )

    def init_weights(self):
        for m in self.modules():
            netutils.init_module(m)

    def forward(self, feature_lrms, pf_uplrms, pf_udpan, pf_hrpan):
        x: list[torch.Tensor] = feature_lrms
        for stage, (f_uplrms, f_udpan, f_hrpan) in enumerate(zip(pf_uplrms[::-1], pf_udpan[::-1], pf_hrpan[::-1])):
            attn = self.__getattr__(f"mha{stage}")(f_uplrms, f_udpan, f_hrpan)
            x = attn + x
            x_res = x
            x = self.__getattr__(f"RB1{stage}")(x)
            x = x + x_res
            if stage <= self.stage_num - 2:
                x_res = F.interpolate(x, scale_factor=(2, 2), mode="bicubic", align_corners=False)
                x = self.__getattr__(f"upsample{stage}")(x)
                x = (x + x_res) / 2
                x = self.__getattr__(f"chdown{stage}")(x)
        x = self.final_proj(x)
        x_bypass = self.conv_bypass(x)
        for i in range(4):
            x = self.RBF[i](x)
        x = self.convF_tail(x) + x_bypass
        return x
