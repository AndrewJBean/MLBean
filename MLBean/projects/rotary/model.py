import torch
from MLBean.data.dataset import TextDataset
from MLBean.modules.model_and_loss import (
  ModelAndOffsetLoss,
  IterModelAndOffsetLoss,
  LossModelWrapper,
)
from MLBean.modules.transformer_modules import RotaryTransformer
from MLBean.modules.iter_transformer import IterTransformerConfig, IterTransformer

from MLBean.projects.rotary.all_config import AllConfig


def build_model_and_loss(config: AllConfig, dataset: TextDataset) -> LossModelWrapper:
  if config.model.rotary is not None:
    return ModelAndOffsetLoss(
      model=RotaryTransformer(
        config=config.model.rotary,
        vocab_size=dataset.vocab_size,
      ),
    )
  elif config.model.iterated is not None:
    return IterModelAndOffsetLoss(
      model=IterTransformer(
        config=config.model.iterated,
        vocab_size=dataset.vocab_size,
      ),
    )
  else:
    raise ValueError("Unsupported config.model")
