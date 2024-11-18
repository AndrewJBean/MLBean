import unittest
import torch

from MLBean.data.dataset import TextDataset, TextPredictionDataset, DatasetConfig


class TestDataset(unittest.TestCase):
  def test_text_dataset(self):
    ds = TextDataset(
      batch_size=4,
      trunc_len=512,
    )

    batch = next(ds)
    self.assertEqual(len(batch), 4)
    self.assertTrue(all([len(row) == 512 for row in batch]))

    test_str = "hello world"
    tokens = ds.strings_to_batch([test_str], trunc_len=512)
    new_str = ds.tokens_to_strings(tokens)
    self.assertEqual(new_str, [test_str])

  def test_text_prediction_dataset(self):
    group_size = 4
    config = DatasetConfig(
      batch_size=4,
      char_map_file="MLBean/data/character_map.txt",
      trunc_len=512,
      group_size=group_size,
      special_tokens_at_end=True,
    )
    ds = TextPredictionDataset.from_config(config)

    batch = next(ds)
    self.assertEqual(len(batch), 2)
    self.assertEqual(len(batch["inputs"]), 4)
    self.assertEqual(len(batch["labels"]), 4)
    self.assertTrue(all([len(row) % group_size == 0 for row in batch["inputs"]]))
    self.assertTrue(all([len(row) % group_size == 0 for row in batch["labels"]]))
    self.assertTrue((batch["inputs"][:, group_size:] == batch["labels"][:, :-group_size]).all())
