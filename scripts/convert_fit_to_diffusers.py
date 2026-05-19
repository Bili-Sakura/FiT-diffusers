#!/usr/bin/env python3
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
