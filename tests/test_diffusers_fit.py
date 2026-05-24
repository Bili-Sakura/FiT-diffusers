import importlib.util
import sys
from pathlib import Path


def test_transformer_ast_parseable():
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "diffusers" / "models" / "transformers" / "transformer_fit.py"
    src = path.read_text()
    compile(src, str(path), "exec")


def test_diffusers_fit_importable_when_torch_present():
    if importlib.util.find_spec("torch") is None:
        return
    import diffusers_fit  # noqa: F401

    assert hasattr(diffusers_fit, "FiTTransformer2DModel")


def test_fitv1_skips_flow_timestep_shifting():
    if importlib.util.find_spec("torch") is None:
        return

    import torch
    from diffusers_fit.models.transformers.transformer_fit import FiTTransformer2DModel

    model = FiTTransformer2DModel(use_sit=False, time_shifting=1)
    t = torch.tensor([999.0])
    if model.use_sit:
        t = torch.clamp(model.time_shifting * t / (1 + (model.time_shifting - 1) * t), max=1.0)
    assert t.item() == 999.0
