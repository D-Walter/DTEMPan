import torch.nn as nn, math, mmengine


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def init_module(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        # variance_scaling_initializer(m.weight)
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ModuleWithInit(nn.Module):
    def __init__(self, **kwargs):
        super(ModuleWithInit, self).__init__(**kwargs)

    def init_weights(self):
        try:
            for m in self.modules:
                if isinstance(m, nn.ModuleList):
                    for sm in m:
                        init_module(sm)
                else:
                    init_module(m)
        except:
            pass


from mmengine.model import BaseModel as mmbm


class BaseModel(mmbm):
    def __init__(self, components: dict, init_cfg, **kwargs):
        super(BaseModel, self).__init__(init_cfg=init_cfg, **kwargs)
        for component_name in components:
            component = components[component_name]
            if component is None:
                continue
            if isinstance(component, nn.Module):
                self.__setattr__(component_name, component)
            elif isinstance(component, dict):
                self.__setattr__(component_name, mmengine.build_from_cfg(component, mmengine.MODELS))
            else:
                raise NotImplementedError()
