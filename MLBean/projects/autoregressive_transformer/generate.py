import signal
import sys
import pathlib

from absl import app, flags

from MLBean.data.dataset import FullExcerptDataset
from MLBean.training.checkpointing import get_latest_checkpoint

from MLBean.projects.autoregressive_transformer.setup_train_dir import get_all_config
from MLBean.projects.autoregressive_transformer.model import build_model_and_loss

import torch


def setup_flags():
  flags.DEFINE_integer("step", None, "The checkpoint step number to load")
  flags.DEFINE_string("dir", None, "The directory to load checkpoints from")
  flags.mark_flag_as_required("dir")


def main(argv):
  if flags.FLAGS.dir is None:
    raise ValueError("Please specify a directory to load checkpoints from")
  chkpt_dir = pathlib.Path(flags.FLAGS.dir)

  dataset = FullExcerptDataset()

  if not (chkpt_dir / "config.json").exists() and not (chkpt_dir / "all_config.json").exists():
    raise ValueError(f"Config file not found in {chkpt_dir}")

  all_config = get_all_config(chkpt_dir)

  model_and_loss = build_model_and_loss(all_config, dataset)
  checkpoint_path = get_latest_checkpoint(chkpt_dir, step=flags.FLAGS.step)
  print(f"Loading checkpoint from {checkpoint_path}")
  model_and_loss.load_state_dict(torch.load(checkpoint_path, weights_only=True))
  model = model_and_loss.model.model
  print(f"Model size: {sum(p.numel() for p in model.parameters()):,}")

  # get a prompt from the user
  prompt = input("Enter a prompt: ")
  batch = dataset.strings_to_batch([prompt])[:, :-1]
  tokens = list(batch[0].cpu().numpy())
  print(prompt, end="")

  max_context_length = 10240
  # max_context_length = 200
  with torch.no_grad():
    model.eval()
    while True:
      batch = torch.tensor(tokens).reshape(1, -1)

      # Ideally, we would maintain state inside the model from previous calls
      # and only compute the layer activations for the new token.
      next_token_logits = model(batch)[:, -1, :]

      temperature = 0.8
      next_token = torch.multinomial(
        torch.softmax(next_token_logits / temperature, dim=-1), num_samples=1
      )
      if next_token in dataset.non_text_tokens:
        break
      tokens.append(next_token)
      if len(tokens) > max_context_length:
        tokens = tokens[-max_context_length:]
      next_char = dataset.tokens_to_string(next_token)[0]
      print(next_char, end="", flush=True)


# handle keyboard interrupt
def signal_handler(sig, frame):
  print("\nKeyboardInterrupt, exiting...")
  sys.exit(0)


if __name__ == "__main__":
  setup_flags()
  signal.signal(signal.SIGINT, signal_handler)
  app.run(main)
