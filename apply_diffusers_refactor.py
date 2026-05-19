#!/usr/bin/env python3
"""Apply FiT -> diffusers-fit refactor. Run from repository root."""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "diffusers_fit"
FIT = ROOT / "fit"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    os.chdir(ROOT)
    if not FIT.exists():
        raise SystemExit(f"Missing {FIT}; run from FiT repo root.")

    print("Creating package layout...")
    for sub in ["models/transformers", "pipelines/fit", "schedulers", "datasets", "utils", "scripts", "tests"]:
        (SRC / sub).mkdir(parents=True, exist_ok=True)

    diffusers_link = ROOT / "src" / "diffusers"
    if diffusers_link.is_symlink() or diffusers_link.exists():
        if diffusers_link.is_symlink():
            diffusers_link.unlink()
        else:
            shutil.rmtree(diffusers_link)
    diffusers_link.symlink_to("diffusers_fit", target_is_directory=True)
    print("  symlink src/diffusers -> diffusers_fit")

    copy_tree(FIT / "scheduler" / "transport", SRC / "schedulers" / "fit_transport")
    copy_tree(FIT / "scheduler" / "improved_diffusion", SRC / "schedulers" / "fit_gaussian_diffusion")
    shutil.copy2(FIT / "data" / "in1k_latent_dataset.py", SRC / "datasets" / "in1k_latent_dataset.py")
    shutil.copy2(FIT / "utils" / "lr_scheduler.py", SRC / "utils" / "lr_scheduler.py")
    shutil.copy2(FIT / "utils" / "sit_eval_utils.py", SRC / "utils" / "sit_eval_utils.py")

    cfg_src = FIT / "utils" / "utils.py"
    cfg_text = cfg_src.read_text(encoding="utf-8")
    if "def default" not in cfg_text:
        cfg_text += '''

def exists(val):
    return val is not None

def isfunction(obj):
    import inspect
    return inspect.isfunction(obj)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
'''
    write(SRC / "utils" / "config.py", cfg_text)
    write(SRC / "utils" / "checkpoint.py", (FIT / "utils" / "eval_utils.py").read_text(encoding="utf-8"))
    write(SRC / "utils" / "__init__.py", "")

    shutil.copy2(FIT / "model" / "rope.py", SRC / "models" / "transformers" / "rope_fit.py")
    shutil.copy2(FIT / "model" / "norms.py", SRC / "models" / "transformers" / "norms_fit.py")
    shutil.copy2(FIT / "model" / "modules.py", SRC / "models" / "transformers" / "modules_fit.py")
    patch_file(
        SRC / "models" / "transformers" / "modules_fit.py",
        [
            ("from fit.model.rope import rotate_half", "from diffusers_fit.models.transformers.rope_fit import rotate_half"),
            ("from fit.model.utils import modulate", ""),
            ("from fit.model.norms import create_norm", "from diffusers_fit.models.transformers.norms_fit import create_norm"),
        ],
    )
    mod = (SRC / "models" / "transformers" / "modules_fit.py").read_text(encoding="utf-8")
    if "def modulate(" not in mod:
        mod = mod.replace(
            "from diffusers_fit.models.transformers.norms_fit import create_norm\n",
            "from diffusers_fit.models.transformers.norms_fit import create_norm\n\n\n"
            "def modulate(x, shift, scale):\n"
            "    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)\n\n",
        )
        (SRC / "models" / "transformers" / "modules_fit.py").write_text(mod, encoding="utf-8")

    tm = (FIT / "model" / "fit_model.py").read_text(encoding="utf-8")
    tm = re.sub(r"from fit\.model\.sincos import.*\n", "", tm)
    tm = tm.replace("from fit.model.", "from diffusers_fit.models.transformers.")
    tm = tm.replace("from fit.utils.eval_utils import init_from_ckpt", "from diffusers_fit.utils.checkpoint import init_from_ckpt")
    header = '''import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from diffusers_fit.models.transformers.modules_fit import (
    PatchEmbedder, TimestepEmbedder, LabelEmbedder, FiTBlock, FinalLayer,
)
from diffusers_fit.utils.checkpoint import init_from_ckpt
from diffusers_fit.models.transformers.rope_fit import VisionRotaryEmbedding


def get_parameter_dtype(parameter: torch.nn.Module):
    params = tuple(parameter.parameters())
    if params:
        return params[0].dtype
    buffers = tuple(parameter.buffers())
    return buffers[0].dtype if buffers else torch.float32


'''
    body = tm.split("class FiT", 1)[1]
    tm = header + "class _FiTCore(nn.Module):" + body + '''

class FiTTransformer2DModel(ModelMixin, ConfigMixin, _FiTCore):
    config_name = "config.json"

    @register_to_config
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dtype(self) -> torch.dtype:
        return get_parameter_dtype(self)


FiT = FiTTransformer2DModel
'''
    write(SRC / "models" / "transformers" / "transformer_fit.py", tm)
    write(SRC / "models" / "transformers" / "__init__.py", 'from .transformer_fit import FiTTransformer2DModel, FiT\n__all__ = ["FiTTransformer2DModel", "FiT"]\n')
    write(SRC / "models" / "__init__.py", "from .transformers import FiTTransformer2DModel\n")
    write(SRC / "datasets" / "__init__.py", "from .in1k_latent_dataset import IN1kLatentDataset, INLatentLoader\n")

    write(
        SRC / "schedulers" / "scheduling_flow_match_fit.py",
        '''from dataclasses import dataclass
from typing import Any, Callable, Optional
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from diffusers_fit.schedulers.fit_transport import Sampler, create_transport

@dataclass
class FiTFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor

class FiTFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(self, path_type="Linear", prediction="velocity", loss_weight=None, sample_eps=None, train_eps=None, snr_type="lognorm", mode="ODE"):
        self.transport = create_transport(path_type=path_type, prediction=prediction, loss_weight=loss_weight, sample_eps=sample_eps, train_eps=train_eps, snr_type=snr_type)
        self.sampler = Sampler(self.transport)
        self.mode = mode
        self._sample_fn = None

    def configure_sampler(self, mode=None, num_steps=250, ode_sampling_method="dopri5", atol=1e-6, rtol=1e-3, reverse=False, sde_sampling_method="Euler", diffusion_form="sigma", diffusion_norm=1.0, last_step="Mean", last_step_size=0.04, **kwargs):
        mode = (mode or self.mode).upper()
        self.mode = mode
        if mode == "ODE":
            self._sample_fn = self.sampler.sample_ode(sampling_method=ode_sampling_method, num_steps=num_steps, atol=atol, rtol=rtol, reverse=reverse)
        else:
            self._sample_fn = self.sampler.sample_sde(sampling_method=sde_sampling_method, diffusion_form=diffusion_form, diffusion_norm=diffusion_norm, last_step=last_step, last_step_size=last_step_size, num_steps=num_steps)
        return self._sample_fn

    def sample(self, latents, model_fn, **model_kwargs):
        if self._sample_fn is None:
            self.configure_sampler()
        return self._sample_fn(latents, model_fn, **model_kwargs)[-1]
''',
    )
    write(SRC / "schedulers" / "__init__.py", "from .scheduling_flow_match_fit import FiTFlowMatchScheduler, FiTFlowMatchSchedulerOutput\n")

    write(
        SRC / "pipelines" / "fit" / "pipeline_fit.py",
        '''from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers_fit.schedulers import FiTFlowMatchScheduler

@dataclass
class FiTPipelineOutput(BaseOutput):
    images: Union[torch.Tensor, List]

class FiTPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(self, transformer, scheduler, vae=None):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor()

    @torch.no_grad()
    def __call__(self, class_labels, height=256, width=256, num_inference_steps=250, guidance_scale=1.0, scale_pow=0.0, sampler_mode="ODE", generator=None, output_type="pil", return_dict=True, **sampler_kwargs):
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype
        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        bs = class_labels.shape[0]
        H, W = height // 8, width // 8
        p = self.transformer.patch_size
        nh, nw = H // p, W // p
        use_sit = self.transformer.use_sit
        cin = self.transformer.in_channels
        shape = (bs, nh * nw, p * p * cin) if use_sit else (bs, p * p * cin, nh * nw)
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        gh = torch.arange(nh, device=device, dtype=torch.long)
        gw = torch.arange(nw, device=device, dtype=torch.long)
        g = torch.meshgrid(gw, gh, indexing="xy")
        grid = torch.cat([g[0].reshape(1, -1), g[1].reshape(1, -1)], dim=0).repeat(bs, 1, 1)
        mask = torch.ones(bs, nh * nw, device=device, dtype=dtype)
        size = torch.tensor((nh, nw), device=device).repeat(bs, 1)[:, None, :]
        if guidance_scale > 1.0:
            latents = torch.cat([latents, latents])
            y = torch.cat([class_labels, torch.full((bs,), self.transformer.num_classes, device=device, dtype=torch.long)])
            grid, mask, size = torch.cat([grid, grid]), torch.cat([mask, mask]), torch.cat([size, size])
            fn = self.transformer.forward_with_cfg
            kw = dict(y=y, grid=grid, mask=mask, size=size, cfg_scale=guidance_scale, scale_pow=scale_pow)
        else:
            fn, kw = self.transformer.forward, dict(y=class_labels, grid=grid, mask=mask, size=size)
        self.scheduler.configure_sampler(mode=sampler_mode, num_steps=num_inference_steps, **sampler_kwargs)
        latents = self.scheduler.sample(latents, fn, **kw)
        if guidance_scale > 1.0:
            latents = latents[:bs]
        latents = self.transformer.unpatchify(latents[..., : nh * nw], (H, W))
        if self.vae is not None:
            image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            image = self.image_processor.postprocess((image / 2 + 0.5).clamp(0, 1), output_type=output_type)
        else:
            image = latents
        if not return_dict:
            return (image,)
        return FiTPipelineOutput(images=image)
''',
    )
    write(SRC / "pipelines" / "fit" / "__init__.py", "from .pipeline_fit import FiTPipeline, FiTPipelineOutput\n")
    write(SRC / "pipelines" / "__init__.py", "from .fit import FiTPipeline\n")
    write(
        SRC / "__init__.py",
        '''from diffusers_fit.models.transformers import FiTTransformer2DModel
from diffusers_fit.pipelines.fit import FiTPipeline, FiTPipelineOutput
from diffusers_fit.schedulers import FiTFlowMatchScheduler
from diffusers_fit.schedulers.fit_transport import create_transport
from diffusers_fit.schedulers.fit_gaussian_diffusion import create_diffusion
__all__ = ["FiTTransformer2DModel", "FiTPipeline", "FiTPipelineOutput", "FiTFlowMatchScheduler", "create_transport", "create_diffusion"]
''',
    )

    def patch_script(name: str, dst: str) -> None:
        text = (ROOT / name).read_text(encoding="utf-8")
        repl = {
            "from fit.schedulers.transport import create_transport": "from diffusers_fit.schedulers.fit_transport import create_transport",
            "from fit.scheduler.transport import create_transport, Sampler": "from diffusers_fit.schedulers.fit_transport import create_transport, Sampler",
            "from fit.schedulers.improved_diffusion import create_diffusion": "from diffusers_fit.schedulers.fit_gaussian_diffusion import create_diffusion",
            "from fit.scheduler.improved_diffusion import create_diffusion": "from diffusers_fit.schedulers.fit_gaussian_diffusion import create_diffusion",
            "from fit.utils.utils import": "from diffusers_fit.utils.config import",
            "from fit.utils.eval_utils import init_from_ckpt": "from diffusers_fit.utils.checkpoint import init_from_ckpt",
            "from fit.utils.eval_utils import": "from diffusers_fit.utils.checkpoint import",
            "from fit.utils.lr_scheduler import get_scheduler": "from diffusers_fit.utils.lr_scheduler import get_scheduler",
            "from fit.utils.sit_eval_utils import": "from diffusers_fit.utils.sit_eval_utils import",
        }
        for a, b in repl.items():
            text = text.replace(a, b)
        write(ROOT / "scripts" / dst, text)

    patch_script("train_fitv2.py", "train_fitv2.py")
    patch_script("train_fit.py", "train_fit.py")
    patch_script("sample_fitv2_ddp.py", "sample_fit_ddp.py")

    write(
        ROOT / "scripts" / "sample_fit.py",
        '''#!/usr/bin/env python3
import argparse
import torch
from diffusers import AutoencoderKL
from diffusers_fit import FiTPipeline, FiTFlowMatchScheduler
from diffusers_fit.utils.checkpoint import init_from_ckpt
from diffusers_fit.utils.config import instantiate_from_config
from omegaconf import OmegaConf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfgdir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--class-label", type=int, default=0)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--num-inference-steps", type=int, default=250)
    p.add_argument("--sampler-mode", default="ODE")
    p.add_argument("--guidance-scale", type=float, default=1.5)
    args = p.parse_args()
    cfg = OmegaConf.load(args.cfgdir)
    model = instantiate_from_config(cfg.diffusion.network_config)
    init_from_ckpt(model, args.ckpt, verbose=True)
    model.eval()
    transport_cfg = OmegaConf.to_container(cfg.diffusion.transport)
    scheduler = FiTFlowMatchScheduler(**transport_cfg)
    vae = AutoencoderKL.from_pretrained(cfg.diffusion.pretrained_first_stage_model_path)
    pipe = FiTPipeline(transformer=model, scheduler=scheduler, vae=vae)
    out = pipe(class_labels=args.class_label, height=args.height, width=args.width, num_inference_steps=args.num_inference_steps, sampler_mode=args.sampler_mode, guidance_scale=args.guidance_scale)
    out.images[0].save("sample.png")
    print("saved sample.png")

if __name__ == "__main__":
    main()
''',
    )

    write(
        ROOT / "scripts" / "convert_fit_to_diffusers.py",
        '''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from diffusers_fit import FiTTransformer2DModel, FiTFlowMatchScheduler
from diffusers_fit.utils.checkpoint import load_state_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    params = dict(cfg.diffusion.network_config.params)
    model = FiTTransformer2DModel(**params)
    sd = load_state_dict(args.checkpoint)
    model.load_state_dict(sd, strict=False)
    out = Path(args.output)
    model.save_pretrained(out / "transformer")
    sched = FiTFlowMatchScheduler(**OmegaConf.to_container(cfg.diffusion.transport))
    sched.save_pretrained(out / "scheduler")
    with open(out / "model_index.json", "w") as f:
        json.dump({"_class_name": "FiTPipeline", "transformer": ["diffusers_fit", "FiTTransformer2DModel"], "scheduler": ["diffusers_fit", "FiTFlowMatchScheduler"], "vae": ["diffusers", "AutoencoderKL"]}, f, indent=2)
    print("saved", out)

if __name__ == "__main__":
    main()
''',
    )

    write(
        ROOT / "pyproject.toml",
        '''[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "diffusers-fit"
version = "0.1.0"
description = "Diffusers-style Flexible Vision Transformer (FiT) components."
readme = "README_DIFFUSERS.md"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
    "diffusers>=0.30.1",
    "torch",
    "safetensors",
    "einops",
    "timm",
    "omegaconf",
    "accelerate",
    "numpy",
    "pillow",
    "torchdiffeq",
]

[project.optional-dependencies]
dev = ["pytest", "wandb", "torchvision"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
''',
    )

    write(
        ROOT / "README_DIFFUSERS.md",
        '''# FiT Diffusers integration

Native [Diffusers](https://github.com/huggingface/diffusers) packaging for FiT, following [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers.git).

## Layout

- `src/diffusers_fit/` — installable package (`pip install -e .`)
- `src/diffusers` → symlink to `diffusers_fit` (upstream copy path)

## Install

```bash
pip install -e .
```

Use `from diffusers_fit import FiTTransformer2DModel, FiTPipeline` for FiT code and `from diffusers import AutoencoderKL` for Hugging Face components.

## Convert & sample

```bash
python scripts/convert_fit_to_diffusers.py --checkpoint CKPT --output OUT --config configs/fitv2/config_fitv2_xl.yaml
python scripts/sample_fit.py --cfgdir configs/fitv2/config_fitv2_xl.yaml --ckpt CKPT --class-label 207
```
''',
    )

    write(
        ROOT / "tests" / "test_fit_diffusers.py",
        '''import pytest
torch = pytest.importorskip("torch")
from diffusers_fit import FiTTransformer2DModel, create_transport
from diffusers_fit.schedulers.fit_transport import Sampler

def test_forward_sit():
    m = FiTTransformer2DModel(context_size=256, patch_size=2, in_channels=4, hidden_size=32, depth=2, num_heads=4, num_classes=10, learn_sigma=False, use_sit=True, online_rope=True)
    x = torch.randn(1, 16, 16)
    out = m(x, torch.tensor([0.5]), torch.tensor([0]), torch.zeros(1, 2, 16).long(), torch.ones(1, 16))
    assert out.shape == x.shape

def test_create_transport():
    t = create_transport(path_type="Linear", prediction="velocity", snr_type="lognorm")
    Sampler(t)
''',
    )

    for yml in (ROOT / "configs").rglob("*.yaml"):
        t = yml.read_text(encoding="utf-8")
        t = t.replace("fit.model.fit_model.FiT", "diffusers_fit.models.transformers.FiTTransformer2DModel")
        t = t.replace("fit.data.in1k_latent_dataset.INLatentLoader", "diffusers_fit.datasets.in1k_latent_dataset.INLatentLoader")
        yml.write_text(t, encoding="utf-8")

    for sh in (ROOT / "tools").glob("*.sh"):
        t = sh.read_text(encoding="utf-8")
        t = t.replace("train_fitv2.py", "scripts/train_fitv2.py").replace("sample_fitv2_ddp.py", "scripts/sample_fit_ddp.py")
        t = t.replace("train_fit.py", "scripts/train_fit.py").replace("sample_fit_ddp.py", "scripts/sample_fit_ddp.py")
        sh.write_text(t, encoding="utf-8")

    for path in [FIT, ROOT / "train_fit.py", ROOT / "train_fitv2.py", ROOT / "sample_fit_ddp.py", ROOT / "sample_fitv2_ddp.py", ROOT / "setup.py"]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        print(f"  removed {path.relative_to(ROOT)}")

    print("Done.")


if __name__ == "__main__":
    main()
