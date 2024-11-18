from MLBean.projects.autoregressive_transformer.all_config import AllConfig
from MLBean.data.dataset import TextPredictionDataset
from MLBean.modules.model_and_loss import ModelAndLoss
from MLBean.modules.transformer_modules import TransformerWrapper


def build_model_and_loss(config: AllConfig, dataset: TextPredictionDataset) -> ModelAndLoss:
  return ModelAndLoss(
    model=TransformerWrapper(
      config=config.model,
      pad_token=dataset.pad_token,
      vocab_size=dataset.vocab_size,
    )
  )
