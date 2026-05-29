#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from diffusers import AutoencoderKL

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:
    safe_load_file = None
    safe_save_file = None

from diffusers_fit.models.transformers.transformer_fit import FiTTransformer2DModel


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = REPO_ROOT.parents[1]

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "FiTv1-XL/2": {
        "context_size": 256,
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": 1152,
        "depth": 28,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "class_dropout_prob": 0.1,
        "num_classes": 1000,
        "learn_sigma": True,
        "use_swiglu": True,
        "use_swiglu_large": True,
        "rel_pos_embed": "rope",
    },
    "FiTv2-XL/2": {
        "context_size": 256,
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": 1152,
        "depth": 36,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "class_dropout_prob": 0.1,
        "num_classes": 1000,
        "learn_sigma": False,
        "use_sit": True,
        "use_swiglu": True,
        "use_swiglu_large": False,
        "q_norm": "layernorm",
        "k_norm": "layernorm",
        "rel_pos_embed": "rope",
        "adaln_type": "lora",
        "adaln_lora_dim": 288,
    },
    "FiTv2-3B/2": {
        "context_size": 256,
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": 2304,
        "depth": 40,
        "num_heads": 24,
        "mlp_ratio": 4.0,
        "class_dropout_prob": 0.1,
        "num_classes": 1000,
        "learn_sigma": False,
        "use_sit": True,
        "use_swiglu": True,
        "use_swiglu_large": False,
        "q_norm": "layernorm",
        "k_norm": "layernorm",
        "rel_pos_embed": "rope",
        "adaln_type": "lora",
        "adaln_lora_dim": 576,
    },
}

DDPM_SCHEDULER_CONFIG = {
    "_class_name": "DDPMScheduler",
    "_diffusers_version": "0.36.0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "clip_sample": False,
    "clip_sample_range": 1.0,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "variance_type": "learned_range",
    "timestep_spacing": "linspace",
    "steps_offset": 0,
    "trained_betas": None,
}

FLOW_MATCH_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.36.0",
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "stochastic_sampling": False,
}

BUNDLE_SCRIPT = REPO_ROOT / "scripts" / "bundle_fit_hub_modules.py"
BUNDLED_TRANSFORMER = REPO_ROOT / "src/diffusers/models/transformers/fit_transformer_2d.py"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; pass --device cpu.")
    return resolved


def _load_state_dict(checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    map_location = device
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to convert .safetensors checkpoints.")
        state_dict = safe_load_file(checkpoint_path, device=str(map_location))
    else:
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "module", "ema"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
    return _clean_state_dict(state_dict)


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("model.", "module.", "transformer.")
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def infer_learn_sigma(state_dict: Dict[str, torch.Tensor], patch_size: int, in_channels: int = 4) -> bool:
    weight = state_dict.get("final_layer.linear.weight")
    if weight is None:
        return True
    base = patch_size * patch_size * in_channels
    return int(weight.shape[0]) == base * 2


def load_imagenet_id2label() -> Dict[int, str]:
    reference_paths = [
        COLLECTION_ROOT / "models/BiliSakura/NiT-diffusers/NiT-XL/model_index.json",
        COLLECTION_ROOT / "models/BiliSakura/DiT-diffusers/DiT-XL-2-256/model_index.json",
        COLLECTION_ROOT / "models/BiliSakura/DiT-MoE-diffusers/DiT-MoE-XL-8E2A/model_index.json",
    ]
    for path in reference_paths:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            id2label = raw.get("id2label")
            if isinstance(id2label, dict):
                return {int(key): value for key, value in id2label.items()}
    raise FileNotFoundError("Could not find a reference model_index.json with ImageNet id2label.")


def _save_config(output_dir: Path, config: Dict[str, Any], filename: str = "config.json"):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / filename, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_weights(output_dir: Path, state_dict: Dict[str, torch.Tensor], safe_serialization: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(state_dict, str(output_dir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def _write_model_index(output_dir: Path, *, model: str, image_size: int):
    id2label_int = load_imagenet_id2label()
    is_fiTv2 = model.startswith("FiTv2")
    model_index = {
        "_class_name": ["pipeline", "FiTv2Pipeline" if is_fiTv2 else "FiTPipeline"],
        "_diffusers_version": "0.36.0",
        "transformer": ["fit_transformer_2d", "FiTTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler" if is_fiTv2 else "DDPMScheduler"],
        "sample_size": int(image_size),
        "id2label": {str(class_id): id2label_int[class_id] for class_id in range(1000)},
    }
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as handle:
        json.dump(model_index, handle, indent=2)
        handle.write("\n")


def _write_readme(output_dir: Path, *, variant_name: str, model: str, image_size: int):
    is_fiTv2 = model.startswith("FiTv2")
    pipeline_class = "FiTv2Pipeline" if is_fiTv2 else "FiTPipeline"
    scheduler_name = "FlowMatchEulerDiscreteScheduler" if is_fiTv2 else "DDPMScheduler"
    sampler_note = "flow matching (velocity ODE)" if is_fiTv2 else "improved diffusion (DDPM respaced)"
    content = f"""---
license: apache-2.0
library_name: diffusers
pipeline_tag: unconditional-image-generation
tags:
  - diffusers
  - fit
  - image-generation
  - class-conditional
  - imagenet
inference: true
---

# {variant_name}

Self-contained Diffusers checkpoint for **{model}**, converted from [`InfImagine/FiT`](https://huggingface.co/InfImagine/FiT).

Each subfolder is a self-contained Diffusers model repo with:

- `model_index.json` (includes ImageNet `id2label`)
- `pipeline.py` (custom `{pipeline_class}`)
- `transformer/fit_transformer_2d.py` and weights
- `scheduler/scheduler_config.json` (`{scheduler_name}`)
- `vae/diffusion_pytorch_model.safetensors`

## Recommended inference ({image_size}×{image_size})

| Setting | Value |
| --- | --- |
| Resolution | {image_size}×{image_size} |
| Sampler | {sampler_note} |
| Steps | 250 |
| CFG scale | 1.5 |
| Dtype | `float32` (or `bfloat16` on Ampere+) |
| VAE | `stabilityai/sd-vae-ft-ema` (bundled under `vae/`) |

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path("./{variant_name}").resolve()
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

print(pipe.id2label[207])
print(pipe.get_label_ids("golden retriever"))

generator = torch.Generator(device="cuda").manual_seed(42)
image = pipe(
    class_labels="golden retriever",
    height={image_size},
    width={image_size},
    num_inference_steps=250,
    guidance_scale=1.5,
    generator=generator,
).images[0]
image.save("demo.png")
```
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def _ensure_bundled_modules() -> None:
    subprocess.run([sys.executable, str(BUNDLE_SCRIPT)], check=True, cwd=REPO_ROOT)


def make_self_contained_repo(output_dir: Path, *, variant_name: str, model: str, image_size: int):
    _ensure_bundled_modules()
    is_fiTv2 = model.startswith("FiTv2")
    pipeline_template = REPO_ROOT / ("templates/pipeline_fiTv2.py" if is_fiTv2 else "templates/pipeline.py")
    shutil.copy2(pipeline_template, output_dir / "pipeline.py")
    if image_size != 256:
        pipeline_text = (output_dir / "pipeline.py").read_text(encoding="utf-8")
        pipeline_text = pipeline_text.replace(
            "DEFAULT_NATIVE_RESOLUTION = 256",
            f"DEFAULT_NATIVE_RESOLUTION = {image_size}",
        )
        (output_dir / "pipeline.py").write_text(pipeline_text, encoding="utf-8")

    transformer_dir = output_dir / "transformer"
    if transformer_dir.exists():
        for path in transformer_dir.glob("*.py"):
            if path.name != "fit_transformer_2d.py":
                path.unlink()
    transformer_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUNDLED_TRANSFORMER, transformer_dir / "fit_transformer_2d.py")

    scheduler_dir = output_dir / "scheduler"
    if scheduler_dir.exists():
        for path in scheduler_dir.glob("*.py"):
            path.unlink()
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    scheduler_config = FLOW_MATCH_SCHEDULER_CONFIG if is_fiTv2 else DDPM_SCHEDULER_CONFIG
    _save_config(scheduler_dir, scheduler_config, filename="scheduler_config.json")

    _write_model_index(output_dir, model=model, image_size=image_size)
    _write_readme(output_dir, variant_name=variant_name, model=model, image_size=image_size)


def _copy_vae(source_vae_dir: Path, target_vae_dir: Path):
    if target_vae_dir.exists():
        shutil.rmtree(target_vae_dir)
    shutil.copytree(source_vae_dir, target_vae_dir)

    safetensors_path = target_vae_dir / "diffusion_pytorch_model.safetensors"
    bin_path = target_vae_dir / "diffusion_pytorch_model.bin"
    if not safetensors_path.exists() and bin_path.exists():
        if safe_save_file is None:
            raise ImportError("Install safetensors to convert bundled VAE weights.")
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
        safe_save_file(state_dict, str(safetensors_path), metadata={"format": "pt"})
        bin_path.unlink()


def _export_vae(vae_id: str, target_vae_dir: Path):
    vae = AutoencoderKL.from_pretrained(vae_id)
    target_vae_dir.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(str(target_vae_dir), safe_serialization=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert original FiT checkpoints to a Diffusers pipeline directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to an original FiT .pt/.bin/.safetensors checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="FiTv1-XL/2")
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--copy-vae", default=None, help="Optional local VAE directory to copy into output/vae.")
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for checkpoint load and --check-load (default: cuda when available).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = _resolve_device(args.device)
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"

    print(f"Loading checkpoint on {device}...")
    state_dict = _load_state_dict(args.checkpoint, device)
    preset = dict(MODEL_PRESETS[args.model])
    if args.learn_sigma is not None:
        preset["learn_sigma"] = args.learn_sigma
    elif "learn_sigma" not in preset:
        preset["learn_sigma"] = infer_learn_sigma(state_dict, patch_size=preset["patch_size"])

    config = {"_class_name": "FiTTransformer2DModel", **preset}
    if args.image_size == 512 and args.model.startswith("FiTv2"):
        patch_grid = args.image_size // 8 // preset["patch_size"]
        config.update(
            {
                "custom_freqs": "ntk-aware",
                "decouple": True,
                "ori_max_pe_len": 16,
                "max_pe_len_h": patch_grid,
                "max_pe_len_w": patch_grid,
                "online_rope": False,
            }
        )

    if args.check_load:
        print(f"Verifying weight load on {device}...")
        model = FiTTransformer2DModel(**preset).to(device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if missing_keys or unexpected_keys:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            raise SystemExit(1)

    save_state_dict = {key: value.detach().cpu() for key, value in state_dict.items()}
    del state_dict
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _save_config(transformer_dir, config)
    _save_weights(transformer_dir, save_state_dict, args.safe_serialization)

    if args.copy_vae is not None:
        _copy_vae(Path(args.copy_vae), output_dir / "vae")
    elif args.vae:
        print(f"Exporting VAE {args.vae}...")
        _export_vae(args.vae, output_dir / "vae")

    variant_name = output_dir.name
    make_self_contained_repo(
        output_dir,
        variant_name=variant_name,
        model=args.model,
        image_size=args.image_size,
    )
    print(f"Saved Diffusers-style FiT pipeline to {output_dir}")


if __name__ == "__main__":
    main()
