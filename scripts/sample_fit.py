#!/usr/bin/env python3
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
