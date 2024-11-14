from typing import List, Tuple, Self, Set, Optional
import os
import platform
from pprint import pprint

import torch
from datasets import load_dataset

from MLBean.data.batch_base import DictionaryBatch
from MLBean.configs.config_base import BaseConfig


class DatasetConfig(BaseConfig):
  batch_size: int
  trunc_len: int
  char_map_file: Optional[str] = None
  group_size: Optional[int] = None
  special_tokens_at_end: Optional[bool] = True


# use ~/huggingface_cache for caching
CACHE_DIR = os.path.expanduser("~/huggingface_cache")


def input_label_split(
  x: torch.Tensor, group_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  x: (batch_size, num_chars)
  """
  if group_size is None:
    group_size = 1
  return x[:, :-group_size], x[:, group_size:]


class FullExcerptDataset(torch.utils.data.IterableDataset):
  OOV = "<OOV>"
  START = "<START>"
  END = "<END>"
  PAD = "<PAD>"

  def __init__(
    self,
    *,
    batch_size: int = 1,
    char_map_file: Optional[str] = None,
    trunc_len: int = 500,
    group_size: Optional[int] = None,
    special_tokens_at_end: Optional[bool] = True,
  ):
    if char_map_file is None:
      this_file_directory = os.path.dirname(os.path.realpath(__file__))
      char_map_file = os.path.join(this_file_directory, "character_map.txt")
    self.batch_size = batch_size
    self.ds = None
    self.trunc_len = trunc_len
    if group_size is None:
      group_size = 1
    self.group_size = group_size
    if self.trunc_len % self.group_size != 0:
      raise ValueError("trunc_len must be divisible by group_size")

    # Load the character mapping
    with open(char_map_file, "r") as f:
      self.tokens = list(f.read())
    self.VOCAB_SIZE = len(self.tokens)
    print("Vocab size:", self.VOCAB_SIZE)
    if special_tokens_at_end:
      self.tokens.extend([self.OOV, self.START, self.END, self.PAD])
    else:
      self.tokens = [self.START, self.END, self.PAD, self.OOV] + self.tokens
    self.token_to_idx = {c: i for i, c in enumerate(self.tokens)}
    print("Character mapping loaded.")

  @staticmethod
  def from_config(config: DatasetConfig) -> Self:
    return FullExcerptDataset(
      batch_size=config.batch_size,
      char_map_file=config.char_map_file,
      trunc_len=config.trunc_len,
      group_size=config.group_size,
      special_tokens_at_end=config.special_tokens_at_end,
    )

  @property
  def vocab_size(self) -> int:
    return len(self.tokens)

  @property
  def oov_token(self) -> int:
    return self.token_to_idx[self.OOV]

  @property
  def start_token(self) -> int:
    return self.token_to_idx[self.START]

  @property
  def end_token(self) -> int:
    return self.token_to_idx[self.END]

  @property
  def pad_token(self) -> int:
    return self.token_to_idx[self.PAD]

  @property
  def non_text_tokens(self) -> Set[int]:
    return {self.start_token, self.end_token, self.oov_token, self.pad_token}

  def __iter__(self) -> Self:
    return self

  def chars_to_tokens(self, chars: List[str]) -> List[int]:
    return (
      [self.start_token]
      + [self.token_to_idx.get(c, self.oov_token) for c in chars]
      + [self.end_token]
    )

  def strings_to_batch(self, strings: List[str], trunc_len: int = 500) -> torch.Tensor:
    batch = []
    for s in strings:
      batch.append(self.chars_to_tokens(list(s)))
    # pad to max_len
    max_len = max(len(row) for row in batch)
    for i in range(len(batch)):
      batch[i] = batch[i] + [self.pad_token] * (max_len - len(batch[i]))

    if max_len > trunc_len:
      for i in range(len(batch)):
        batch[i] = batch[i][:trunc_len]
    elif max_len % self.group_size != 0:
      # truncate to multiple of group_size
      for i in range(len(batch)):
        batch[i] = batch[i][: -(max_len % self.group_size)]

    return torch.tensor(batch, dtype=torch.long)

  def __next__(self) -> DictionaryBatch:
    if self.ds is None:
      self.ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "CC-MAIN-2013-20",
        streaming=platform.system() != "Darwin",
        cache_dir=CACHE_DIR,
        download_mode="reuse_dataset_if_exists",
      )
      self.ds = iter(self.ds["train"])

    records = [list(next(self.ds)["text"]) for _ in range(self.batch_size)]
    batch = self.strings_to_batch(records, trunc_len=self.trunc_len)

    inputs, labels = input_label_split(batch, group_size=self.group_size)
    return DictionaryBatch({"inputs": inputs, "labels": labels})

  def tokens_to_string(self, tokens: torch.Tensor) -> str:
    # assuming 2D batch tensor
    return [
      "".join([self.tokens[tok] for tok in row if tok not in self.non_text_tokens])
      for row in tokens.tolist()
    ]

  def to_dataloader(self) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(self, batch_size=None)


if __name__ == "__main__":
  this_file_directory = os.path.dirname(os.path.realpath(__file__))
  char_map_file = os.path.join(this_file_directory, "character_map.txt")
  ds = FullExcerptDataset(batch_size=5, char_map_file=char_map_file)
  for i, batch in enumerate(ds):
    pprint(ds.tokens_to_string(batch["inputs"]))
    if i > 3:
      break
