from dataclasses import dataclass
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
