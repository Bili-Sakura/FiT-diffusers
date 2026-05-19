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
