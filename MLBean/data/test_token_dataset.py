import json
import math
import pathlib
import tempfile
import unittest

import numpy as np

from MLBean.data.token_dataset import TokenDataset, mix_stride
from MLBean.util.pretokenize import write_shard


class TestTokenDataset(unittest.TestCase):
  def test_mix_stride_permutes(self):
    for n in [1, 2, 3, 7, 10, 100, 1000]:
      k = mix_stride(n)
      self.assertEqual(sorted((i * k) % n for i in range(n)), list(range(n)))
    # Large n: just check coprimality (full permute is too big to materialize).
    for n in [100_000_007, 100_000_008]:
      k = mix_stride(n)
      self.assertEqual(math.gcd(k, n), 1)
      self.assertTrue(1 <= k < n)

  def test_batches_are_permuted_windows(self):
    seq_len = 4
    # 10 sequences worth of tokens plus a short remainder that must be dropped.
    tokens = np.arange(10 * seq_len + 3, dtype=np.dtype("<u2"))
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      write_shard(root / "shard_000000.bin", tokens)
      with open(root / "metadata.json", "w") as f:
        json.dump(
          {
            "tokenizer": "gpt2",
            "vocab_size": 50257,
            "eot_token": 50256,
            "dtype": "<u2",
            "dataset": "test",
            "dataset_config": "test",
            "total_tokens": int(tokens.size),
            "shards": [{"file": "shard_000000.bin", "num_tokens": int(tokens.size)}],
          },
          f,
        )
      ds = TokenDataset(batch_size=2, seq_len=seq_len, source=root)
      batch = next(iter(ds))
      self.assertEqual(tuple(batch.shape), (2, seq_len))

      n = 10
      k = mix_stride(n)
      expected0 = torch_row(tokens, (0 * k) % n, seq_len)
      expected1 = torch_row(tokens, (1 * k) % n, seq_len)
      self.assertTrue(torch_equal(batch[0], expected0))
      self.assertTrue(torch_equal(batch[1], expected1))

      # start=1 skips the first batch (sequences i=0,1), so first yield is i=2,3.
      ds_start = TokenDataset(batch_size=2, seq_len=seq_len, source=root, start=1)
      batch1 = next(iter(ds_start))
      expected2 = torch_row(tokens, (2 * k) % n, seq_len)
      expected3 = torch_row(tokens, (3 * k) % n, seq_len)
      self.assertTrue(torch_equal(batch1[0], expected2))
      self.assertTrue(torch_equal(batch1[1], expected3))


def torch_row(tokens: np.ndarray, seq_idx: int, seq_len: int) -> np.ndarray:
  start = seq_idx * seq_len
  return np.asarray(tokens[start : start + seq_len], dtype=np.int64)


def torch_equal(t, expected: np.ndarray) -> bool:
  return bool((t.numpy() == expected).all())
