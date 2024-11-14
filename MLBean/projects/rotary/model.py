from MLBean.projects.rotary.all_config import AllConfig
from MLBean.data.dataset import FullExcerptDataset
from MLBean.modules.model_and_loss import ModelAndLoss
from MLBean.modules.transformer_modules import DictBatchWrapper, RotaryTransformer


def build_model_and_loss(config: AllConfig, dataset: FullExcerptDataset) -> ModelAndLoss:
  if config.model.rotary is not None:
    return ModelAndLoss(
      model=DictBatchWrapper(
        model=RotaryTransformer(
          config=config.model.rotary,
          vocab_size=dataset.vocab_size,
        ),
      )
    )
  else:
    raise ValueError("Unsupported config.model")
