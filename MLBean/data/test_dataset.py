import unittest

from MLBean.data.dataset import TextDataset, TextPredictionDataset, DatasetConfig


class TestDataset(unittest.TestCase):
  def test_strings_to_batch_round_trip(self):
    ds = TextDataset(batch_size=1, trunc_len=512)
    tokens = ds.strings_to_batch(["hello world"], trunc_len=512)
    self.assertEqual(ds.tokens_to_strings(tokens), ["hello world"])

  def test_prediction_label_shift(self):
    group_size = 4
    ds = TextPredictionDataset.from_config(
      DatasetConfig(
        batch_size=1,
        char_map_file="MLBean/data/character_map.txt",
        trunc_len=32,
        group_size=group_size,
      )
    )
    # Avoid FineWeb: build a batch from a local string through the parent helpers.
    batch = ds.strings_to_batch(["a" * 32], trunc_len=32)
    batch = batch[:, : (batch.size(1) // group_size) * group_size]
    from MLBean.data.dataset import input_label_split

    inputs, labels = input_label_split(batch, group_size=group_size)
    self.assertTrue((inputs[:, group_size:] == labels[:, :-group_size]).all())
