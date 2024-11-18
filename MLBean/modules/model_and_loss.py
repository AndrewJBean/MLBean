from typing import Dict
import torch


class ModelAndLoss(torch.nn.Module):
  def __init__(self, *, model: torch.nn.Module):
    super().__init__()
    self.model = model
    self.loss = torch.nn.CrossEntropyLoss()

  def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    x:
      {
        "inputs": (batch_size, sequence_length),
        "labels": (batch_size, sequence_length),
      }
    Returns:
      {
        "logits": (batch_size, sequence_length, char_vocab_size),
        "loss": torch.Tensor,
        "labels": torch.Tensor,
      }
    """
    output = self.model(x)
    # logits: (batch_size, sequence_length, char_vocab_size)
    logits = output["logits"]
    # squash first 2 dimensions
    y = logits.reshape(-1, logits.size(-1))
    labels = x["labels"]
    loss = self.loss(y, labels.flatten())
    return {"loss": loss, "labels": labels, **output}


class ModelAndOffsetLoss(torch.nn.Module):
  def __init__(self, *, model: torch.nn.Module, offset: int = 1):
    """
    Args:
      model: torch.nn.Module where input is tokens: (batch_size, sequence_length)
             and output is logits: (batch_size, sequence_length, char_vocab_size)
      offset: int, how many characters ahead to predict, default 1
    """
    super().__init__()
    self._model = model
    self.loss = torch.nn.CrossEntropyLoss()
    self.offset = offset

  @property
  def model(self):
    return self._model

  def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    x.size() == (batch_size, sequence_length)

    return dict must have key "loss", to conform to
      MLBean.training.trainer.Trainer
    "labels" is optional, but many metrics require it
    """

    inputs = x[:, : -self.offset]
    labels = x[:, self.offset :]
    logits = self.model(inputs)
    y = logits.reshape(-1, logits.size(-1))
    loss = self.loss(y, labels.flatten())
    return {"loss": loss, "labels": labels, "logits": logits}
