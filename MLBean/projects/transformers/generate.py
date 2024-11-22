import signal
import sys
import pathlib

from absl import app, flags

from MLBean.data.dataset import TextDataset
from MLBean.training.checkpointing import get_latest_checkpoint

from MLBean.projects.transformers.all_config import get_all_config
from MLBean.projects.transformers.model import build_model_and_loss

import torch

FLAGS = flags.FLAGS


def setup_flags():
  flags.DEFINE_integer("step", None, "The checkpoint step number to load")
  flags.DEFINE_string("dir", None, "The directory to load checkpoints from")
  flags.mark_flag_as_required("dir")


def main(argv):
  if FLAGS.dir is None:
    raise ValueError("Please specify a directory to load checkpoints from")
  chkpt_dir = pathlib.Path(FLAGS.dir)
  all_config = get_all_config(chkpt_dir, maybe_create=False)

  dataset = TextDataset.from_config(all_config.dataset_train)
  model_and_loss = build_model_and_loss(all_config, dataset)
  checkpoint_path = get_latest_checkpoint(chkpt_dir, step=FLAGS.step)
  print(f"Loading checkpoint from {checkpoint_path}")
  model_and_loss.load_state_dict(torch.load(checkpoint_path, weights_only=True))
  model = model_and_loss.model
  print(f"Model size: {sum(p.numel() for p in model.parameters()):,}")

  # get a prompt from the user
  prompt = input("Enter a prompt: ")
  # cut off <END> token
  batch = dataset.strings_to_batch([prompt])[:, :-1]
  tokens = list(batch[0].cpu().numpy())
  print(prompt, end="")

  max_context_length = 1024
  with torch.no_grad():
    model.eval()
    while True:
      batch = torch.tensor(tokens).reshape(1, -1)

      # Ideally, we would maintain state inside the model from previous calls
      # and only compute the layer activations for the new token.
      all_logits = model(batch)
      next_token_logits = all_logits[:, -1, :]
      temperature = 0.8
      next_token = torch.multinomial(
        torch.softmax(next_token_logits / temperature, dim=-1), num_samples=1
      )
      if next_token in dataset.non_text_tokens:
        break
      tokens.append(next_token)
      if len(tokens) > max_context_length:
        tokens = tokens[-max_context_length:]
      next_char = dataset.tokens_to_strings(next_token)[0]
      print(next_char, end="", flush=True)


# handle keyboard interrupt
def signal_handler(sig, frame):
  print("\nKeyboardInterrupt, exiting...")
  sys.exit(0)


if __name__ == "__main__":
  setup_flags()
  signal.signal(signal.SIGINT, signal_handler)
  app.run(main)
