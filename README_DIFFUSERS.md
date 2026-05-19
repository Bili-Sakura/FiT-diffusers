# FiT Diffusers integration

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
