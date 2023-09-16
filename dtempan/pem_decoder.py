import torch, torch.nn as nn, mmengine, methods.models.dtempan.netutils as netutils, torch.nn.functional as F
from .netutils import fill_up_weights
from .basic_blocks import ResBlock
from INADToolBox.attentions import SpatialAttention, SpectralAttention, CoorAttention, CrossAttention

try:
    from mmcv.ops import DeformConv2dPack as DCN
except:
    DCN = torch.nn.Conv2d
    print("No DCN")


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deform_groups=1)
        self.actf = nn.Sequential(nn.BatchNorm2d(cho, momentum=0.1), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


@mmengine.MODELS.register_module()
class TAAS(nn.Module):
    def __init__(self, in_chs, out_ch, up_factors):
        super(TAAS, self).__init__()
        self.out_ch = out_ch
        for i in range(len(in_chs)):
            c = in_chs[i]
            f = int(up_factors[i])
            setattr(self, f"proj_{str(i)}", DeformConv(c, out_ch))
            up = nn.ConvTranspose2d(out_ch, out_ch, f * 2, stride=f, padding=f // 2, output_padding=0, groups=out_ch, bias=False) if up_factors[i] > 1 else nn.Conv2d(out_ch, out_ch, 1)
            fill_up_weights(up)
            setattr(self, f"up_{str(i)}", up)
            setattr(self, f"cbam_{str(i)}", nn.Sequential(SpatialAttention(), SpectralAttention(out_ch)))
            setattr(self, f"node_{str(i)}", DeformConv(out_ch, out_ch))

    def forward(self, features):
        tar_shape = features[0].shape
        tar_shape = (tar_shape[0], 64, tar_shape[2], tar_shape[3])
        output = torch.zeros(tar_shape).to(features[0].device)
        for i in range(0, len(features)):
            project = getattr(self, f"proj_{str(i)}")
            upsample = getattr(self, f"up_{str(i)}")
            upsample = upsample(project(features[i]))
            upsample = getattr(self, f"cbam_{str(i)}")(upsample)
            node = getattr(self, f"node_{str(i)}")
            output = node(output + upsample)
        return output


@mmengine.MODELS.register_module()
class EdgeDetectionAttention(nn.Module):
    def __init__(self, ch_pan, ch_feat, ch_ms):
        super(EdgeDetectionAttention, self).__init__()
        self.pan_ch_align = nn.Sequential(nn.Conv2d(ch_pan, ch_feat, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True))
        self.feat_ch_align = nn.Sequential(nn.Conv2d(ch_feat, ch_feat, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True))
        self.w1 = nn.Parameter(torch.ones(1) * 0.5)
        self.w2 = nn.Parameter(torch.ones(1) * 0.5)
        w = 6
        h = 10
        self.avg_pool = nn.AdaptiveAvgPool2d((w, h))

        self.c_attention1 = nn.Sequential(nn.Conv2d(ch_feat * 2, ch_feat * 2, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True))
        self.c_attention2 = nn.Sequential(nn.Conv2d(ch_feat * 2, ch_feat * 2, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True))

        self.final_pan_ch_align = nn.Sequential(
            nn.Conv2d(ch_feat * 2, ch_feat, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True)
        )
        self.final_feat_ch_align = nn.Sequential(
            nn.Conv2d(ch_feat * 2, ch_feat, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True)
        )

        self.cross_attention = CrossAttention(ch_feat, ch_feat)
        self.coor_attention = CoorAttention(ch_feat, ch_feat)

        self.final_proj = nn.Sequential(
            ResBlock(ch_feat * 2, ch_feat, activate=nn.GELU),
            nn.Conv2d(ch_feat, ch_feat // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=ch_feat // 2),
            nn.Conv2d(ch_feat // 2, ch_ms, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, pan, feat):
        pan = self.pan_ch_align(pan)
        feat = self.feat_ch_align(feat)

        x = torch.cat([pan, feat], dim=1)
        b, c, h, w = x.size()
        y = self.avg_pool(x)

        y_t1 = self.c_attention1(y)
        y_t2 = self.c_attention2(y)
        bs, c, h, w = y_t1.shape
        y_t1 = y_t1.view(bs, c, h * w)
        y_t2 = y_t2.view(bs, c, h * w)

        y_t1_T = y_t1.permute(0, 2, 1)
        y_t2_T = y_t2.permute(0, 2, 1)
        M_t1 = torch.matmul(y_t1, y_t1_T)
        M_t2 = torch.matmul(y_t2, y_t2_T)
        M_t1 = F.softmax(M_t1, dim=-1)
        M_t2 = F.softmax(M_t2, dim=-1)

        M_s1 = torch.matmul(y_t1, y_t2_T)
        M_s2 = torch.matmul(y_t2, y_t1_T)
        M_s1 = F.softmax(M_s1, dim=-1)
        M_s2 = F.softmax(M_s2, dim=-1)

        x_t1 = x
        x_t2 = x
        bs, c, h, w = x_t1.shape
        x_t1 = x_t1.contiguous().view(bs, c, h * w)
        x_t2 = x_t2.contiguous().view(bs, c, h * w)
        x_t1 = torch.matmul(self.w1 * M_t1 + (1 - self.w1) * M_s1, x_t1).contiguous().view(bs, c, h, w)
        x_t2 = torch.matmul(self.w2 * M_t2 + (1 - self.w2) * M_s2, x_t2).contiguous().view(bs, c, h, w)

        x_t1 = self.final_pan_ch_align(x_t1)
        x_t2 = self.final_feat_ch_align(x_t2)
        ca = self.cross_attention(x_t1, x_t2)
        pan = self.coor_attention(pan)
        edge = torch.cat([ca, pan], dim=1)
        edge = self.final_proj(edge)
        return edge
    
@mmengine.MODELS.register_module()
class EdgeDetectionAttentionV2(nn.Module):
    def __init__(self, ch_pan, ch_feat, ch_ms):
        super(EdgeDetectionAttentionV2, self).__init__()
        self.pan_ch_align = nn.Sequential(nn.Conv2d(ch_pan, ch_feat, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True))
        self.feat_ch_align = nn.Sequential(nn.Conv2d(ch_feat, ch_feat, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True))
        self.w1 = nn.Parameter(torch.ones(1) * 0.5)
        self.w2 = nn.Parameter(torch.ones(1) * 0.5)
        w = 6
        h = 10
        self.avg_pool = nn.AdaptiveAvgPool2d((w, h))

        self.c_attention1 = nn.Sequential(nn.Conv2d(ch_feat, ch_feat, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True))
        self.c_attention2 = nn.Sequential(nn.Conv2d(ch_feat, ch_feat, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True))

        self.final_pan_ch_align = nn.Sequential(
            nn.Conv2d(ch_feat, ch_feat, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True)
        )
        self.final_feat_ch_align = nn.Sequential(
            nn.Conv2d(ch_feat, ch_feat, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(num_features=ch_feat), nn.LeakyReLU(0.3, inplace=True)
        )

        self.cross_attention = CrossAttention(ch_feat, ch_feat)
        self.coor_attention = CoorAttention(ch_feat, ch_feat)

        self.final_proj = nn.Sequential(
            ResBlock(ch_feat * 2, ch_feat, activate=nn.GELU),
            nn.Conv2d(ch_feat, ch_feat // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=ch_feat // 2),
            nn.Conv2d(ch_feat // 2, ch_ms, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, pan, feat):
        pan = self.pan_ch_align(pan)
        feat = self.feat_ch_align(feat)

        x = pan + feat
        b, c, h, w = x.size()
        y = self.avg_pool(x)

        y_t1 = self.c_attention1(y)
        y_t2 = self.c_attention2(y)
        bs, c, h, w = y_t1.shape
        y_t1 = y_t1.view(bs, c, h * w)
        y_t2 = y_t2.view(bs, c, h * w)

        y_t1_T = y_t1.permute(0, 2, 1)
        y_t2_T = y_t2.permute(0, 2, 1)
        M_t1 = torch.matmul(y_t1, y_t1_T)
        M_t2 = torch.matmul(y_t2, y_t2_T)
        M_t1 = F.softmax(M_t1, dim=-1)
        M_t2 = F.softmax(M_t2, dim=-1)

        M_s1 = torch.matmul(y_t1, y_t2_T)
        M_s2 = torch.matmul(y_t2, y_t1_T)
        M_s1 = F.softmax(M_s1, dim=-1)
        M_s2 = F.softmax(M_s2, dim=-1)

        x_t1 = pan
        x_t2 = feat
        bs, c, h, w = x_t1.shape
        x_t1 = x_t1.contiguous().view(bs, c, h * w)
        x_t2 = x_t2.contiguous().view(bs, c, h * w)
        x_t1 = torch.matmul(self.w1 * M_t1 + (1 - self.w1) * M_s1, x_t1).contiguous().view(bs, c, h, w) + pan
        x_t2 = torch.matmul(self.w2 * M_t2 + (1 - self.w2) * M_s2, x_t2).contiguous().view(bs, c, h, w) + feat

        x_t1 = self.final_pan_ch_align(x_t1)
        x_t2 = self.final_feat_ch_align(x_t2)
        ca = self.cross_attention(x_t1, x_t2)
        pan = self.coor_attention(pan)
        edge = torch.cat([ca, pan], dim=1)
        edge = self.final_proj(edge)
        return edge


@mmengine.MODELS.register_module()
class PreciseEdgeMaintainingDecoder(netutils.BaseModel):
    def __init__(self, components, init_cfg=None):
        assert False not in [n in components.keys() for n in ["fagg", "attn"]], "Missing component."
        super().__init__(components, init_cfg=init_cfg)

    def forward(self, x_hrpan, pf_hrpan, mode="predict"):
        agg_feat = self.fagg(pf_hrpan)
        edge = self.attn(x_hrpan, agg_feat)
        return edge
