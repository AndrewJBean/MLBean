import signal
import sys
import pathlib
from absl import app, flags
import os

from MLBean.data.dataset import TextPredictionDataset
from MLBean.environment.devices import get_device
from MLBean.training.metrics import TopKAccuracy, Loss
from MLBean.training.optimizer import build_optimizer
from MLBean.training.trainer import Trainer

from MLBean.projects.conv_pooled.all_config import get_all_config
from MLBean.projects.conv_pooled.model import build_model_and_loss


FLAGS = flags.FLAGS


def setup_flags():
  flags.DEFINE_string("jobdir", None, "The directory to load checkpoints from")
  flags.mark_flag_as_required("jobdir")


def main(argv):
  device = get_device()
  chkpt_dir = pathlib.Path(FLAGS.jobdir)
  chkpt_dir.mkdir(parents=True, exist_ok=True)
  os.chdir(chkpt_dir)

  all_config = get_all_config(chkpt_dir)

  dataset = TextPredictionDataset.from_config(all_config.dataset_train)
  model_and_loss = build_model_and_loss(all_config, dataset)
  print(f"Model size: {sum(p.numel() for p in model_and_loss._model.model.parameters()):,}")
  optimizer = build_optimizer(params=model_and_loss.parameters(), config=all_config.optimizer)
  metrics = dict(
    loss=Loss(),
    **{f"accuracy(k={k})": TopKAccuracy(k=k) for k in [1, 2, 3]},
  )

  trainer = Trainer(
    config=all_config.training,
    model_and_loss=model_and_loss,
    optimizer=optimizer,
    metrics=metrics,
    device=device,
  )
  trainer.train(dataset)


# handle keyboard interrupt
def signal_handler(sig, frame):
  print("\nKeyboardInterrupt, exiting...")
  sys.exit(0)


if __name__ == "__main__":
  setup_flags()
  # signal.signal(signal.SIGINT, signal_handler)
  app.run(main)
