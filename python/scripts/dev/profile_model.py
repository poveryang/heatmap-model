import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def profile_model():
    from thop import profile
    from thop import clever_format
    from hmap.model import CSPPAFPNNet

    model = CSPPAFPNNet(
        in_channels=1,
        out_channels=9,
        object_classes=3,
        geometry_channels=6,
        base_channels=24,
        depth=2,
        head_channels=48,
    )
    in_tensor = torch.randn(1, 1, 960, 640)
    flops, params, ret_dict = profile(model, inputs=(in_tensor,), ret_layer_info=True)
    flops, params = clever_format([flops, params], "%.3f")
    ret_dict = {k: clever_format([v[0], v[1]], "%.3f") for k, v in ret_dict.items()}
    print("|{:-^15}|{:-^15}|{:-^15}|".format("Layer", "FLOPS", "Params"))
    print("|{:^15}|{:^15}|{:^15}|".format("Total", flops, params))
    for k, v in ret_dict.items():
        print("|{:^15}|{:^15}|{:^15}|".format(k, v[0], v[1]))


if __name__ == '__main__':
    import torch
    profile_model()
