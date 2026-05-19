# Diffusers-style FiT layout

This repository follows the same layout as [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers): implementation code lives under `src/diffusers`, while the **installable Python package name** is `diffusers_fit` so it does not replace the upstream `diffusers` distribution on your environment.

## Install

```bash
pip install diffusers torch torchvision accelerate safetensors omegaconf einops timm torchdiffeq tqdm pillow numpy
pip install -e .
```

Import the FiT transformer and flow helpers:

```python
from diffusers_fit import FiTTransformer2DModel, create_transport, Sampler, FiTFlowPipeline
from diffusers import AutoencoderKL
```

`FiTTransformer2DModel` subclasses Hugging Face `ModelMixin` / `ConfigMixin` and keeps the original FiT state dict layout, so existing checkpoints still load with `init_from_ckpt` or `model.load_state_dict` after you strip any extra keys.

`FiTFlowPipeline` expects a pre-built ODE/SDE `sample_fn` from `Sampler(transport).sample_ode(...)` (same pattern as the legacy sampling scripts) plus an SD VAE.

## Upstreaming

To propose these modules inside `huggingface/diffusers`, copy the tree under `src/diffusers` into the matching paths in the upstream repo and register the classes in the lazy-import tables, as described in the NiT project README.
