import sys
import pathlib

from absl import app, flags

from MLBean.data.dataset import TextDataset
from MLBean.modules.model_and_loss import ModelAndOffsetLoss, ModelAndLoss
from MLBean.modules.transformer_modules import RotaryTransformer, DictBatchWrapper

from MLBean.projects.rotary.all_config import get_all_config

import torch

FLAGS = flags.FLAGS


def setup_flags():
  flags.DEFINE_string("src", None, "dir with checkpoints to convert")
  flags.DEFINE_string("dst", None, "dir to save converted checkpoints")
  flags.mark_flag_as_required("src")
  flags.mark_flag_as_required("dst")


def main(argv):
  if FLAGS.src is None:
    raise ValueError("Please specify a directory to load checkpoints from")
  if FLAGS.dst is None:
    raise ValueError("Please specify a directory to save converted checkpoints to")
  src_dir = pathlib.Path(FLAGS.src)
  dst_dir = pathlib.Path(FLAGS.dst)
  dst_dir.mkdir(parents=True, exist_ok=True)
  all_config = get_all_config(src_dir, maybe_create=False)

  # copy the config file to the new directory
  src_config = src_dir / "all_config.json"
  dst_config = dst_dir / "all_config.json"
  dst_config.write_text(src_config.read_text())

  # copy metrics.jsonl
  src_metrics = src_dir / "metrics.jsonl"
  dst_metrics = dst_dir / "metrics.jsonl"
  dst_metrics.write_text(src_metrics.read_text())

  all_checkpoints = list(src_dir.glob("checkpoint_*.pt"))
  for checkpoint_path in all_checkpoints:
    print(f"Converting {checkpoint_path}")
    dst_checkpoint_path = dst_dir / checkpoint_path.name

    dataset = TextDataset.from_config(all_config.dataset_train)
    # this was the old way to make a module(input)->loss
    model_and_loss = ModelAndLoss(
      model=DictBatchWrapper(
        model=RotaryTransformer(
          config=all_config.model.rotary,
          vocab_size=dataset.vocab_size,
        ),
      )
    )
    print(f"Loading checkpoint from {checkpoint_path}")
    model_and_loss.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model = model_and_loss._model.model
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,}")

    model_and_loss = ModelAndOffsetLoss(model=model)

    # save the model
    print(f"Saving to {dst_checkpoint_path}...", end="")
    torch.save(model_and_loss.state_dict(), dst_checkpoint_path)
    print("done")


if __name__ == "__main__":
  setup_flags()
  app.run(main)
