import torch.optim as optim, math

_base_ = ["base/base_quickbird.py"]
custom_imports = dict(
    imports=_base_.custom_imports["imports"]
    + [
        "methods.models.dtempan.attentions",
        "methods.models.dtempan.fusion",
        "methods.models.dtempan.mstm_decoder",
        "methods.models.dtempan.pem_decoder",
        "methods.models.dtempan.parc_convnext",
        "methods.models.dtempan.pem_decoder",
        "methods.models.dtempan.dtem_transformer",
    ],
    allow_failed_imports=False,
)
backbone_target_ch = 256
stage_num = 3
ms_dims = [64, 128, backbone_target_ch]
hr_patch_size = 256
optim_wrapper = dict(optimizer=dict(type=optim.AdamW, lr=0.001))
# !!!WARNING: Please modify the following content carefully!!!
ms_ch = _base_.ms_ch
pan_ch = _base_.pan_ch
factor = _base_.factor
lr_patch_size = hr_patch_size // factor
hr_pf_factor = [(math.pow(2, i)) for i in range(stage_num)]
hr_pf_rs = [int(_base_.hr_patch_size // f) for f in hr_pf_factor]
components = {
    "encoder_lrms_u": dict(type="ParC_ConvNeXt", depths=[2, 4, 2], dims=ms_dims, ParC_insert_locs=[1, 2, 1], stages_rs=hr_pf_rs, in_chans=ms_ch),
    # "encoder_lrms_u":dict(type="LFE", in_channels=4),
    "encoder_lrpan_u": dict(type="LFE", in_channels=1),
    "encoder_hrpan": dict(type="ParC_ConvNeXt", depths=[2, 4, 2], dims=ms_dims, ParC_insert_locs=[1, 2, 1], stages_rs=hr_pf_rs, in_chans=pan_ch),
    "backbone_lrms": dict(type="SFE", in_feats=ms_ch, num_res_blocks=3, n_feats=backbone_target_ch),
    "mstm_decoder": dict(type="MultiscaleTextureMaintainingDecoder", ms_ch=ms_ch, ms_dims=ms_dims, pan_dims=ms_dims, num_res_blocks=[2, 2, 2]),
    "pem_decoder": dict(
        type="PreciseEdgeMaintainingDecoder",
        components=dict(
            fagg=dict(type="TAAS", in_chs=ms_dims, out_ch=64, up_factors=hr_pf_factor),
            attn=dict(type="EdgeDetectionAttention", ch_pan=1, ch_feat=64, ch_ms=4),
        ),
    ),
    "fusion": dict(type="TextureEdgeFusion"),
}

model = dict(type="DTEMTransformer", components=components, factor=int(_base_.factor))
