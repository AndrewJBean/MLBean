import torch
from MLBean.data.dataset import TextDataset
from MLBean.modules.model_and_loss import ModelAndOffsetLoss
from MLBean.modules.transformer_modules import RotaryTransformer

from MLBean.projects.rotary.all_config import AllConfig


def build_model_and_loss(config: AllConfig, dataset: TextDataset) -> torch.nn.Module:
  if config.model.rotary is not None:
    return ModelAndOffsetLoss(
      model=RotaryTransformer(
        config=config.model.rotary,
        vocab_size=dataset.vocab_size,
      ),
    )
  else:
    raise ValueError("Unsupported config.model")
