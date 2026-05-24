# Copyright 2026 FiT diffusers port.

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor

from ...models.transformers.transformer_fit import FiTTransformer2DModel


@dataclass
class FiTPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]


DEFAULT_NATIVE_RESOLUTION = 256

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> model_dir = Path("./FiTv1-XL-2-256").resolve()
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe.to("cuda")

        >>> print(pipe.id2label[207])
        >>> print(pipe.get_label_ids("golden retriever"))

        >>> generator = torch.Generator(device="cuda").manual_seed(42)
        >>> image = pipe(
        ...     class_labels="golden retriever",
        ...     height=256,
        ...     width=256,
        ...     num_inference_steps=250,
        ...     guidance_scale=1.5,
        ...     generator=generator,
        ... ).images[0]
        ```
"""


class FiTPipeline(DiffusionPipeline):
    r"""
    Class-conditional FiTv1 pipeline using improved-diffusion sampling and an SD VAE decoder.

    FiT uses a flexible token layout (patch sequence + 2D grid) rather than a fixed latent grid.
    """

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer: FiTTransformer2DModel,
        vae=None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
        null_class_id: Optional[int] = None,
        diffusion_config: Optional[Dict[str, object]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae)
        self.image_processor = VaeImageProcessor()
        if diffusion_config is None:
            diffusion_config = {
                "noise_schedule": "linear",
                "use_kl": False,
                "sigma_small": False,
                "predict_xstart": False,
                "learn_sigma": True,
                "rescale_learned_sigmas": False,
                "diffusion_steps": 1000,
            }
        self.register_to_config(diffusion_config=diffusion_config)

        if null_class_id is None:
            null_class_id = int(getattr(self.transformer.config, "num_classes", 1000))
        self.register_to_config(null_class_id=int(null_class_id))

        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)

    @property
    def vae_scale_factor(self) -> int:
        if self.vae is None:
            return 8
        block_out_channels = getattr(self.vae.config, "block_out_channels", None)
        if block_out_channels:
            return int(2 ** (len(block_out_channels) - 1))
        return 8

    @staticmethod
    def _normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
        if not id2label:
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
        label2id: Dict[str, int] = {}
        for class_id, value in id2label.items():
            for synonym in value.split(","):
                synonym = synonym.strip()
                if synonym:
                    label2id[synonym] = int(class_id)
        return dict(sorted(label2id.items()))

    @property
    def id2label(self) -> Dict[int, str]:
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        labels = [label] if isinstance(label, str) else label
        if not self.labels:
            raise ValueError("No id2label mapping is available in this checkpoint.")
        missing = [item for item in labels if item not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown labels: {missing}. Example valid labels: {preview}, ...")
        return [self.labels[item] for item in labels]

    def _normalize_class_labels(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.Tensor],
    ) -> List[int]:
        if isinstance(class_labels, torch.Tensor):
            class_labels = class_labels.detach().cpu().tolist()
        if isinstance(class_labels, int):
            return [class_labels]
        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)
        if not class_labels:
            raise ValueError("`class_labels` cannot be empty.")
        if isinstance(class_labels[0], str):
            return self.get_label_ids(class_labels)  # type: ignore[arg-type]
        return [int(class_id) for class_id in class_labels]  # type: ignore[union-attr]

    @staticmethod
    def _load_create_diffusion():
        try:
            from ...schedulers.improved_diffusion.fit_improved_sampler import create_diffusion
        except ImportError:
            from fit_improved_sampler import create_diffusion
        return create_diffusion

    def _build_diffusion(self, num_inference_steps: int):
        create_diffusion = self._load_create_diffusion()
        cfg = dict(self.config.diffusion_config)
        cfg["timestep_respacing"] = str(num_inference_steps)
        return create_diffusion(**cfg)

    @staticmethod
    def _prepare_fit_inputs(
        batch_size: int,
        n_patch_h: int,
        n_patch_w: int,
        patch_size: int,
        in_channels: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        return randn_tensor(
            (batch_size, (patch_size**2) * in_channels, n_patch_h * n_patch_w),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _prepare_grid_mask_size(
        batch_size: int,
        n_patch_h: int,
        n_patch_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_h = torch.arange(n_patch_h, dtype=torch.long, device=device)
        grid_w = torch.arange(n_patch_w, dtype=torch.long, device=device)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.cat([grid[0].reshape(1, -1), grid[1].reshape(1, -1)], dim=0).repeat(batch_size, 1, 1)
        mask = torch.ones(batch_size, n_patch_h * n_patch_w, device=device, dtype=dtype)
        size = torch.tensor((n_patch_h, n_patch_w), device=device, dtype=torch.long).repeat(batch_size, 1)[:, None, :]
        return grid, mask, size

    @torch.inference_mode()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.Tensor] = 207,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 250,
        guidance_scale: float = 1.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[FiTPipelineOutput, Tuple]:
        r"""
        Generate class-conditional samples from a FiTv1 checkpoint.

        Examples:
            <!-- this section is replaced by replace_example_docstring -->
        """
        class_labels_list = self._normalize_class_labels(class_labels)
        batch_size = len(class_labels_list)
        native_size = DEFAULT_NATIVE_RESOLUTION
        height = native_size if height is None else int(height)
        width = native_size if width is None else int(width)

        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {self.vae_scale_factor}, got ({height}, {width})."
            )
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError(f"Unsupported `output_type`: {output_type}")

        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        patch_size = int(self.transformer.config.patch_size)
        n_patch_h, n_patch_w = latent_h // patch_size, latent_w // patch_size

        z = self._prepare_fit_inputs(
            batch_size,
            n_patch_h,
            n_patch_w,
            patch_size,
            int(self.transformer.in_channels),
            device,
            model_dtype,
            generator,
        )
        grid, mask, size = self._prepare_grid_mask_size(batch_size, n_patch_h, n_patch_w, device, model_dtype)
        class_labels_tensor = torch.tensor(class_labels_list, device=device, dtype=torch.long)

        using_cfg = guidance_scale > 1.0
        if using_cfg:
            z = torch.cat([z, z], dim=0)
            y_null = torch.full((batch_size,), int(self.config.null_class_id), device=device, dtype=torch.long)
            y = torch.cat([class_labels_tensor, y_null], dim=0)
            grid = torch.cat([grid, grid], dim=0)
            mask = torch.cat([mask, mask], dim=0)
            size = torch.cat([size, size], dim=0)
            model_kwargs = dict(y=y, grid=grid, mask=mask, size=size, cfg_scale=guidance_scale)
            sample_fn = self.transformer.forward_with_cfg
        else:
            model_kwargs = dict(y=class_labels_tensor, grid=grid, mask=mask, size=size)
            sample_fn = self.transformer.forward

        diffusion = self._build_diffusion(num_inference_steps)
        samples = diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=self.progress_bar is not None,
            device=device,
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = samples[..., : n_patch_h * n_patch_w]
        samples = self.transformer.unpatchify(samples, (latent_h, latent_w))

        if self.vae is not None:
            samples = self.vae.decode(samples / self.vae.config.scaling_factor).sample
            samples = self.image_processor.postprocess(samples, output_type=output_type)
        elif output_type != "latent":
            raise ValueError("Cannot decode latents without a VAE.")

        self.maybe_free_model_hooks()
        if not return_dict:
            return (samples,)
        return FiTPipelineOutput(images=samples)


__all__ = ["FiTPipeline", "FiTPipelineOutput"]
