import torch
from MLBean.data.dataset import TextDataset
from MLBean.modules.model_and_loss import (
  ModelAndOffsetLoss,
  IterModelAndOffsetLoss,
  LossModelWrapper,
)
from MLBean.modules.transformer_modules import (
  RotaryTransformer,
  AutoregressiveTransformer,
)
from MLBean.modules.iter_transformer import IterTransformer

from MLBean.projects.transformers.all_config import AllConfig


def build_model_and_loss(config: AllConfig, dataset: TextDataset) -> LossModelWrapper:
  if config.model.rotary is not None:
    return ModelAndOffsetLoss(
      model=RotaryTransformer(
        config=config.model.rotary,
        vocab_size=dataset.vocab_size,
      ),
    )
  elif config.model.basic is not None:
    return ModelAndOffsetLoss(
      model=AutoregressiveTransformer(
        config=config.model.basic,
        vocab_size=dataset.vocab_size,
        pad_token=dataset.pad_token,
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
