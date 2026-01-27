from .clip_models import ClipModel


VALID_NAMES = {
    # Provide neutral CLIP baselines (paths should be user-provided/local)
    'CLIP:ViT-B/16': 'openai/clip-vit-base-patch16',
    'CLIP:ViT-B/32': 'openai/clip-vit-base-patch32',
    'CLIP:ViT-L/14': 'openai/clip-vit-large-patch14',
}


def get_model(name, opt):
    assert name in VALID_NAMES.keys(), f"Unknown arch '{name}'. Supported: {list(VALID_NAMES.keys())}"
    if name.startswith("CLIP:"):
        return ClipModel(VALID_NAMES[name], opt)
    raise ValueError(f"Unsupported arch: {name}")
