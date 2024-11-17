import unittest
import torch

from MLBean.modules.transformer_modules import (
  MLPConfig,
  RotaryAttentionConfig,
  RotaryEncoderBlockConfig,
  RotaryTransformerConfig,
  RotaryTransformer,
  AutoregressiveTransformerConfig,
  SinusoidalPositionalEncodingConfig,
  PositionalEncodingConfig,
  MultiHeadAttentionConfig,
  EncoderBlockConfig,
  AutoregressiveTransformer,
)


class TestModules(unittest.TestCase):
  EMB_DIMS = 1024
  NUM_HEADS = 16
  NUM_LAYERS = 12
  VOCAB_SIZE = 1000
  BATCH_SHAPE = (4, 512)

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
    m = RotaryTransformer(config=config, vocab_size=self.VOCAB_SIZE)
    self.assertTrue(isinstance(m, torch.nn.Module))

    fake_input = torch.randint(0, self.VOCAB_SIZE, self.BATCH_SHAPE)
    out = m(fake_input)
    self.assertEqual(out.shape, self.BATCH_SHAPE + (self.VOCAB_SIZE,))

  def get_transformer_config(self):
    return AutoregressiveTransformerConfig(
      embedding_dims=self.EMB_DIMS,
      positional_encoding=PositionalEncodingConfig(
        sinusoidal=SinusoidalPositionalEncodingConfig(
          relative_freq_spacing=1.2,
          base_freq=1.0,
        ),
      ),
      encoder_block=EncoderBlockConfig(
        multi_head_attention=MultiHeadAttentionConfig(num_heads=self.NUM_HEADS),
        mlp=MLPConfig(layer_dims=[self.EMB_DIMS, self.EMB_DIMS]),
      ),
      num_layers=self.NUM_LAYERS,
    )

  def test_transformer_config(self):
    config = self.get_transformer_config()
    self.assertEqual(config.embedding_dims, self.EMB_DIMS)
    self.assertEqual(config.positional_encoding.sinusoidal.relative_freq_spacing, 1.2)
    self.assertEqual(config.positional_encoding.sinusoidal.base_freq, 1.0)
    self.assertEqual(config.encoder_block.multi_head_attention.num_heads, self.NUM_HEADS)
    self.assertEqual(config.encoder_block.mlp.layer_dims, [self.EMB_DIMS, self.EMB_DIMS])
    self.assertEqual(config.num_layers, self.NUM_LAYERS)

  def test_transformer_model(self):
    config = self.get_transformer_config()
    m = AutoregressiveTransformer(config=config, vocab_size=self.VOCAB_SIZE)
    self.assertTrue(isinstance(m, torch.nn.Module))

    fake_input = torch.randint(0, self.VOCAB_SIZE, self.BATCH_SHAPE)
    out = m(fake_input)
    self.assertEqual(out.shape, self.BATCH_SHAPE + (self.VOCAB_SIZE,))


if __name__ == "__main__":
  unittest.main()
