import os
import json
from typing import Tuple, Dict, Optional
import datetime
import torch
import torcheval.metrics

from MLBean.configs.config_base import BaseConfig
from MLBean.training.checkpointing import get_checkpoint_info, save_checkpoint, restore_checkpoint
from MLBean.modules.model_and_loss import LossModelWrapper


LABEL_KEY = "labels"
METRICS_JSON_LINES = "metrics.jsonl"


class CheckpointingConfig(BaseConfig):
  interval: int
  # if None, use current directory
  directory: Optional[str] = None


class EvalConfig(BaseConfig):
  steps: int
  interval: int


class TrainerConfig(BaseConfig):
  num_steps: int
  log_interval: int
  eval: Optional[EvalConfig] = None
  checkpointing: Optional[CheckpointingConfig] = None


def input_label_split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  x: (batch_size, num_chars)
  """
  return x[:, :-1], x[:, 1:]


def batch_to_device_iter_wrapper(ds_iter: torch.utils.data.IterableDataset, device: torch.device):
  for x in ds_iter:
    yield x.to(device)


class Trainer:
  def __init__(
    self,
    config: TrainerConfig,
    model_and_loss: LossModelWrapper,
    optimizer: torch.optim.Optimizer,
    metrics: Dict[str, torcheval.metrics.Metric] = {},
    device: Optional[torch.device] = None,
  ):
    self.num_steps = config.num_steps
    self.log_interval = config.log_interval
    self.config = config
    self.metrics = metrics
    self.model_and_loss = model_and_loss
    self.optimizer = optimizer
    self.device = device
    self.checkpoint_dir = None
    if self.config.checkpointing is not None:
      self.checkpoint_dir = self.config.checkpointing.directory
    if self.checkpoint_dir is None:
      self.checkpoint_dir = os.getcwd()
    os.makedirs(self.checkpoint_dir, exist_ok=True)
    if self.device is not None:
      print(f"Using device: {self.device}")
      for metric in self.metrics.values():
        metric.to(self.device)
      self.model_and_loss.to(self.device)

  def train(
    self,
    dataset: torch.utils.data.IterableDataset,
    eval_dataset: Optional[torch.utils.data.IterableDataset] = None,
  ):
    steps_already_done = self.maybe_restore_checkpoint()
    if steps_already_done >= self.num_steps:
      print("Training already completed.")
      return

    ds_iter = batch_to_device_iter_wrapper(iter(dataset), self.device)
    eval_ds_iter = (
      batch_to_device_iter_wrapper(iter(eval_dataset), self.device)
      if eval_dataset is not None
      else ds_iter
    )

    loss_accum = 0.0
    loss_count = 0
    train_time_accum = datetime.timedelta(0)
    train_steps_count = 0
    for step in range(1 + steps_already_done, 1 + self.num_steps):
      start_time = datetime.datetime.now()
      x = next(ds_iter)
      loss = self.train_step(x)
      loss_accum += loss.item()
      loss_count += 1
      train_steps_count += 1
      train_time_accum += datetime.datetime.now() - start_time

      if step % self.log_interval == 0:
        steps_per_sec = train_steps_count / train_time_accum.total_seconds()
        avg_loss = loss_accum / loss_count
        ts = datetime.datetime.now()
        metrics = {
          "step": step,
          "steps_per_sec": steps_per_sec,
          "ts": ts.timestamp(),
          "loss": avg_loss,
        }
        self.log_metrics({"split": "train", "metrics": metrics})
        print(
          f"train: Step={step}, "
          f"ts={ts.isoformat()}, "
          f"Steps/s={steps_per_sec:.2f}, "
          f"Loss={avg_loss}"
        )
        train_time_accum = datetime.timedelta(0)
        train_steps_count = 0
        loss_accum = 0.0
        loss_count = 0

      self.maybe_save_checkpoint(step)
      self.maybe_eval(step, eval_ds_iter)

  def maybe_eval(self, step: int, ds_iter: torch.utils.data.IterableDataset):
    if self.config.eval is None or not self.metrics or step % self.config.eval.interval != 0:
      return

    metrics_start_time = datetime.datetime.now()
    metrics = self.compute_metrics(ds_iter)
    metrics_end_time = datetime.datetime.now()
    time_difference = metrics_end_time - metrics_start_time
    steps_per_sec = self.config.eval.steps / time_difference.total_seconds()
    eval_print_components = [f"eval: Step={step}, Steps/s={steps_per_sec:.2f}"] + [
      f"{name}={value}" for name, value in metrics.items()
    ]
    joiner = ", " if len(eval_print_components) < 4 else "\n  "
    print(joiner.join(eval_print_components))
    metrics = {
      "step": step,
      "steps_per_sec": steps_per_sec,
      "ts": datetime.datetime.now().timestamp(),
      **{name: value.item() for name, value in metrics.items()},
    }
    self.log_metrics({"split": "eval", "metrics": metrics})

  def log_metrics(self, metrics):
    metrics_jsonl_path = os.path.join(self.checkpoint_dir, METRICS_JSON_LINES)
    with open(metrics_jsonl_path, "a") as f:
      f.write(json.dumps(metrics) + "\n")

  def maybe_save_checkpoint(self, step: int):
    if self.config.checkpointing is None or step % self.config.checkpointing.interval != 0:
      return
    save_checkpoint(model=self.model_and_loss, step=step, checkpoint_dir=self.checkpoint_dir)

  def maybe_restore_checkpoint(self) -> int:
    """
    return step number of the checkpoint restored
    """
    if self.config.checkpointing is None:
      return 0
    return restore_checkpoint(model=self.model_and_loss, checkpoint_dir=self.checkpoint_dir)

  def train_step(self, x: torch.Tensor) -> torch.Tensor:
    self.model_and_loss.train()
    self.optimizer.zero_grad()
    loss = self.model_and_loss(x)["loss"]
    loss.backward()
    self.optimizer.step()
    return loss

  def compute_metrics(self, ds_iter: torch.utils.data.IterableDataset):
    self.model_and_loss.eval()
    results = {}
    for name, metric in self.metrics.items():
      metric.reset()
    for _ in range(self.config.eval.steps):
      x = next(ds_iter)
      with torch.no_grad():
        y = self.model_and_loss(x)
      for name, metric in self.metrics.items():
        metric.update(y)
    for name, metric in self.metrics.items():
      results[name] = metric.compute()
    return results
