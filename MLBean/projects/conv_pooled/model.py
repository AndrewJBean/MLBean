from MLBean.projects.conv_pooled.all_config import AllConfig
from MLBean.data.dataset import FullExcerptDataset
from MLBean.modules.model_and_loss import ModelAndLoss
from MLBean.modules.conv_pooled_transformer import ConvPooledTransformerWrapper


def build_model_and_loss(config: AllConfig, dataset: FullExcerptDataset) -> ModelAndLoss:
  if config.model.conv_pooled_transformer is not None:
    return ModelAndLoss(
      model=ConvPooledTransformerWrapper(
        config=config.model.conv_pooled_transformer,
        vocab_size=dataset.vocab_size,
      )
    )
  else:
    raise ValueError("Unsupported config.model")
