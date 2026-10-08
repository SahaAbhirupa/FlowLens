from .baselines import ConditionalGAN, ConditionalVAE, SegmentationCNN
from .flow import FlowLens

MODELS = {
    "flow": FlowLens,
    "vae": ConditionalVAE,
    "gan": ConditionalGAN,
    "segmentation": SegmentationCNN,
}

# per-model learning rates used in the experiments
DEFAULT_LR = {"flow": 5e-4, "vae": 1e-3, "gan": 2e-4, "segmentation": 1e-3}


def build_model(name: str, n_cells: int, in_ch: int = 3, **kwargs):
    return MODELS[name](n_cells=n_cells, in_ch=in_ch, **kwargs)
