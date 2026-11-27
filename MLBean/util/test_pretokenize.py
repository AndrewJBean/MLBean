import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

from MLBean.util import pretokenize


class TestPretokenize(unittest.TestCase):
  def test_shard_round_trip(self):
    """Wrong dtype/endianness would silently corrupt every training window."""
    for dtype in [np.dtype("<u2"), np.dtype("<u4")]:
      tokens = np.arange(1000, dtype=dtype)
      with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "shard.bin"
        pretokenize.write_shard(path, tokens)
        loaded = pretokenize.read_shard(path, dtype)
        np.testing.assert_array_equal(loaded, tokens)

  def test_tokenize_docs_prepends_eot_to_each_doc(self):
    """Document packing depends on a leading EOT, including for empty docs."""
    encoder = tiktoken.get_encoding("gpt2")
    texts = ["hello world", "", "second document"]
    tokens = pretokenize.tokenize_docs(encoder, texts, num_threads=2)
    self.assertEqual(int((tokens == encoder.eot_token).sum()), len(texts))
    self.assertEqual(encoder.decode(tokens.tolist()).split("<|endoftext|>"), [""] + texts)

  def test_progress_round_trip(self):
    """Resume must restore cursor, shards, and unflushed buffer exactly."""
    dtype = np.dtype("<u2")
    fingerprint = {
      "tokenizer": "gpt2",
      "dtype": dtype.str,
      "dataset": "HuggingFaceFW/fineweb-edu",
      "dataset_config": "CC-MAIN-2013-20",
      "shard_size": 100,
    }
    with tempfile.TemporaryDirectory() as tmp:
      out_dir = pathlib.Path(tmp) / "gpt2"
      out_dir.mkdir()
      pretokenize.write_shard(out_dir / "shard_000000.bin", np.arange(100, dtype=dtype))
      buffer = np.empty(100, dtype=dtype)
      buffer[:30] = np.arange(100, 130, dtype=dtype)
      shards = [{"file": "shard_000000.bin", "num_tokens": 100}]
      pretokenize.save_progress(
        out_dir,
        docs_seen=12,
        parquet_index=2,
        parquet_row=50,
        total_tokens=130,
        shards=shards,
        buffer=buffer,
        fill=30,
        fingerprint=fingerprint,
      )
      loaded = pretokenize.load_progress(out_dir, fingerprint, dtype, shard_size=100, files=[])
      (
        docs_seen,
        parquet_index,
        parquet_row,
        total_tokens,
        loaded_shards,
        loaded_buffer,
        fill,
      ) = loaded
      self.assertEqual(
        (docs_seen, parquet_index, parquet_row, total_tokens, fill), (12, 2, 50, 130, 30)
      )
      self.assertEqual(loaded_shards, shards)
      np.testing.assert_array_equal(loaded_buffer[:fill], buffer[:fill])

  def test_iter_text_batches_resumes_mid_file(self):
    """Resume must seek to a parquet row, not rescan earlier documents."""
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      files = []
      for i, prefix in enumerate(["f0", "f1"]):
        path = root / f"part-{i}.parquet"
        pq.write_table(pa.table({"text": [f"{prefix}-{j}" for j in range(5)]}), path)
        files.append(path.name)

      with mock.patch.object(
        pretokenize, "ensure_local_parquet", side_effect=lambda _ds, name: str(root / name)
      ):
        batches = list(
          pretokenize.iter_text_batches("ds", files, start_file=1, start_row=3, batch_size=2)
        )
      self.assertEqual([t for _f, _r, batch in batches for t in batch], ["f1-3", "f1-4"])
