from typing import Optional
import pathlib
from dataclasses import dataclass
import torch


@dataclass
class CheckpointInfo:
  step: int
  path: pathlib.Path


def path_to_step(path: pathlib.Path) -> int:
  """
  Extract the step number from the checkpoint path.
  Assumes that the step number is the last sequence of digits before a ".pt" extension.
  """

  trimmed = str(path).split(".pt")[0]
  pos = len(trimmed) - 1
  while pos >= 0 and trimmed[pos].isdigit():
    pos -= 1
  if pos == len(trimmed) - 1:
    raise ValueError(
      f"Could not extract step number from path: {path}. "
      "Expected a sequence of digits before the '.pt' extension"
    )
  return int(trimmed[pos + 1 :])


def get_checkpoint_info(chkpt_dir: str, step: Optional[int] = None) -> Optional[CheckpointInfo]:
  chkpt_dir = pathlib.Path(chkpt_dir)
  checkpoints = list(chkpt_dir.glob("*.pt"))
  if len(checkpoints) == 0:
    return None
  step_to_chkpt = {path_to_step(chkpt): chkpt for chkpt in checkpoints}
  if step is not None:
    if step in step_to_chkpt:
      return CheckpointInfo(step=step, path=step_to_chkpt[step])
    return None
  max_step = max(step_to_chkpt.keys())
  max_path = step_to_chkpt[max_step]
  return CheckpointInfo(step=max_step, path=max_path)


def save_checkpoint(model: torch.nn.Module, step: int, checkpoint_dir: str):
  dest = pathlib.Path(checkpoint_dir) / f"checkpoint_{step:010d}.pt"
  print(f"Saving checkpoint to {dest} ...... ", end="")
  torch.save(model.state_dict(), dest)
  print("done")


def restore_checkpoint(
  model: torch.nn.Module, checkpoint_dir: str, step: Optional[int] = None
) -> int:
  """
  Restore the model from the latest checkpoint in the given directory.
  If step is provided, restore the model from the checkpoint with the given step.

  Returns the step number of the checkpoint restored.

  If no checkpoint is found, returns 0.

  Args:
    model: The model to restore the checkpoint to
    checkpoint_dir: The directory to load the checkpoint from
    step: The step number of the checkpoint to restore.
          If None (default), restore the latest checkpoint

  Returns:
    The step number of the checkpoint, or 0 if no checkpoint is found
  """

  checkpoint_info = get_checkpoint_info(checkpoint_dir, step)
  if checkpoint_info is None:
    return 0
  latest_checkpoint = checkpoint_info.step
  checkpoint_path = checkpoint_info.path
  print(f"Restoring checkpoint from {checkpoint_path}")
  model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
  return latest_checkpoint
