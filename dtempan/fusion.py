import mmengine
from mmengine.model import BaseModel
from .netutils import ModuleWithInit


@mmengine.MODELS.register_module()
class TextureEdgeFusion(BaseModel, ModuleWithInit):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

    def forward(self, lrms_u, texture, edge, bit_max):
        assert texture is not None or edge is not None
        if texture is None:
            res = edge
        elif edge is None:
            res = texture
        else:
            res = edge + texture
        hrms = lrms_u - res
        return hrms, res
