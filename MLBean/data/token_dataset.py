"""Deterministic packed-token dataset over pretokenize shards."""

from __future__ import annotations

import json
import math
import pathlib
from typing import Iterator, List, Tuple, Union

import numpy as np
import torch

from MLBean.util.pretokenize import read_shard

PathLike = Union[str, pathlib.Path]

# Large prime used as a multiplicative stride. For each file we take it mod N_file and
# bump until coprime so i |-> (i * K) % N_file permutes {0..N_file-1}.
MIX_PRIME = 100_000_007


def mix_stride(n_sequences: int, preferred: int = MIX_PRIME) -> int:
  """Returns K in 1..n_sequences-1 with gcd(K, n_sequences) == 1 (or 1 if n_sequences <= 1)."""
  if n_sequences <= 1:
    return 1
  k = preferred % n_sequences
  if k == 0:
    k = 1
  while math.gcd(k, n_sequences) != 1:
    k += 1
    if k >= n_sequences:
      return 1
  return k


class TokenDataset(torch.utils.data.IterableDataset):
  """
  Reads flat token shards from a pretokenize output directory and yields batches of shape
  [batch_size, seq_len].

  Files are visited in metadata order. Within a file of usable length
  N_file = len(tokens) // seq_len (remainder dropped), the i-th emitted window is the
  contiguous block starting at ((i * K) % N_file) * seq_len, with K = mix_stride(N_file).
  After the last file, iteration restarts from the first (needed by the training loop).

  ``start`` skips the first ``start`` batches (i.e. ``start * batch_size`` sequences)
  on the first pass only, for resume.
  """

  def __init__(
    self, batch_size: int, seq_len: int, source: PathLike, start: int = 0
  ):
    if batch_size < 1:
      raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if seq_len < 1:
      raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if start < 0:
      raise ValueError(f"start must be >= 0, got {start}")
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.start = start
    self.source = pathlib.Path(source).expanduser()
    metadata_path = self.source / "metadata.json"
    if not metadata_path.exists():
      raise FileNotFoundError(f"Missing {metadata_path}")
    with open(metadata_path) as f:
      metadata = json.load(f)
    self.dtype = np.dtype(metadata["dtype"])
    self.vocab_size = int(metadata["vocab_size"])
    self.eot_token = int(metadata["eot_token"])
    self.shards: List[Tuple[pathlib.Path, int]] = [
      (self.source / shard["file"], int(shard["num_tokens"]))
      for shard in metadata["shards"]
    ]
    if not self.shards:
      raise ValueError(f"No shards listed in {metadata_path}")

  def __iter__(self) -> Iterator[torch.Tensor]:
    skip_sequences = self.start * self.batch_size
    batch: List[torch.Tensor] = []
    while True:
      for path, num_tokens in self.shards:
        n_sequences = num_tokens // self.seq_len
        if n_sequences == 0:
          continue
        if skip_sequences >= n_sequences:
          skip_sequences -= n_sequences
          continue
        tokens = read_shard(path, self.dtype)
        stride = mix_stride(n_sequences)
        for i in range(skip_sequences, n_sequences):
          offset = ((i * stride) % n_sequences) * self.seq_len
          row = np.asarray(tokens[offset : offset + self.seq_len], dtype=np.int64)
          batch.append(torch.from_numpy(row))
          if len(batch) == self.batch_size:
            yield torch.stack(batch, dim=0)
            batch = []
        skip_sequences = 0
