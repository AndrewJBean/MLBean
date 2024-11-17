import unittest
import torch

from MLBean.modules.transformer_modules import (
  MLPConfig,
  RotaryAttentionConfig,
  RotaryEncoderBlockConfig,
  RotaryTransformerConfig,
  RotaryTransformer,
)


class TestModules(unittest.TestCase):
  EMB_DIMS = 1024
  NUM_HEADS = 16
  NUM_LAYERS = 12

  def get_rotary_config(self):
    return RotaryTransformerConfig(
      embedding_dims=self.EMB_DIMS,
      encoder_block=RotaryEncoderBlockConfig(
        attention=RotaryAttentionConfig(
          num_heads=self.NUM_HEADS,
          min_theta=1.0,
          max_theta=1000.0,
        ),
        mlp=MLPConfig(layer_dims=[self.EMB_DIMS, self.EMB_DIMS]),
      ),
      num_layers=self.NUM_LAYERS,
    )

  def test_rotary_config(self):
    config = self.get_rotary_config()
    self.assertEqual(config.embedding_dims, self.EMB_DIMS)
    self.assertEqual(config.encoder_block.attention.num_heads, self.NUM_HEADS)
    self.assertEqual(config.encoder_block.attention.min_theta, 1.0)
    self.assertEqual(config.encoder_block.attention.max_theta, 1000.0)
    self.assertEqual(config.encoder_block.mlp.layer_dims, [self.EMB_DIMS, self.EMB_DIMS])
    self.assertEqual(config.num_layers, self.NUM_LAYERS)

  def test_rotary_model(self):
    config = self.get_rotary_config()
    m = RotaryTransformer(config=config, vocab_size=1000)
    self.assertTrue(isinstance(m, torch.nn.Module))

    fake_input = torch.randint(0, 1000, (4, 512))
    out = m(fake_input)
    self.assertEqual(out.shape, (4, 512, 1000))


if __name__ == "__main__":
  unittest.main()
