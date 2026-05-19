from dataclasses import dataclass
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
