from typing import Optional, Tuple
from MLBean.configs.config_base import BaseConfig, UnionLikeConfig

import torch


class AdamConfig(BaseConfig):
  """
  torch.optim.Adam
  """

  lr: float = 0.001
  betas: Tuple[float, float] = (0.9, 0.999)
  eps: float = 1e-08
  weight_decay: float = 0.0


class AdamWConfig(AdamConfig):
  """
  torch.optim.AdamW
  """

  lr: float = 0.001
  betas: Tuple[float, float] = (0.9, 0.999)
  eps: float = 1e-08
  weight_decay: float = 0.01


class OptimizerConfig(UnionLikeConfig):
  adam: Optional[AdamConfig] = None
  adamw: Optional[AdamWConfig] = None


def build_optimizer(params, config: OptimizerConfig):
  if config.adam is not None:
    kwargs = config.adam.model_dump()
    return torch.optim.Adam(params, **kwargs)
  if config.adamw is not None:
    kwargs = config.adamw.model_dump()
    return torch.optim.AdamW(params, **kwargs)
  raise ValueError("OptimizerConfig not recognized")
