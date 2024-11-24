"""
Transformer language model with input convolution
followed by pooling and transformer layers.

Outputs will predict groups of next-token logits.
Let the group size be `G` and the number of tokens
be `N = G * Ng` where `Ng` is the number groups in
the sequence.
input: (Bsz, Ni)
embedded characters: (Bsz, Ni, De)
permute to: (Bsz, De, Ni)
left pad with (Ni % G) + Kc - 1 zeros: (Bsz, De, N + Kc-1)
conv layer: kernel size `Kc` (already left zero-padded)
            provides a causal convolution
conv output: (Bsz, Dc, N)
activation: ReLU -> (Bsz, Dc, N)
pooling: (max or mean) kernel size `Kp` and stride `G`
         first pad left with `Kp - 1` -inf values
"""

from typing import Optional
import math

import torch

from MLBean.configs.config_base import UnionLikeConfig, BaseConfig
from MLBean.data.batch_base import DictionaryBatch
from MLBean.modules.transformer_modules import (
  PositionalEncodingConfig,
  EncoderBlockConfig,
  PositionalEncoding,
  EncoderBlock,
  MLPConfig,
  MLP,
)


def pad_left(
  x: torch.Tensor, pad_size: int, pad_value: float | torch.Tensor = -math.inf
) -> torch.Tensor:
  """
  x: (Bsz, D, N)
  output: (Bsz, D, N + pad_size)
  """
  Bsz, D, N = x.size()
  pad = torch.full((Bsz, D, pad_size), pad_value, dtype=x.dtype, device=x.device)
  return torch.cat([pad, x], dim=-1)


def left_padded_pooling(x: torch.Tensor, pooling: torch.nn.Module) -> torch.Tensor:
  """
  x: (Bsz, N, D)
  """
  x = pad_left(x, pooling.kernel_size - 1)
  return pooling(x)


class InputConvPoolConfig(BaseConfig):
  """
  Configuration for the input convolution and pooling.
  """

  group_size: int
  conv_kernel_size: int
  conv_out_channels: int
  activation: str = "relu"
  pooling_kernel_size: int
  pooling_type: str = "max"


class InputConvPool(torch.nn.Module):
  """
  Input convolution and pooling module.
  char embedding and positional encoding are
  expected to be done before this module.
  """

  def __init__(self, *, config: InputConvPoolConfig, emb_dims: int):
    super().__init__()
    self.config = config
    self.conv_layer = torch.nn.Conv1d(
      in_channels=emb_dims,
      out_channels=config.conv_out_channels,
      kernel_size=config.conv_kernel_size,
    )
    if config.activation == "relu":
      self.activation = torch.nn.ReLU()
    else:
      raise ValueError(f"Unknown activation: {config.activation}")
    if config.pooling_type == "max":
      self.pooling = torch.nn.MaxPool1d(
        kernel_size=config.pooling_kernel_size,
        stride=config.group_size,
      )
    elif config.pooling_type == "mean":
      self.pooling = torch.nn.AvgPool1d(
        kernel_size=config.pooling_kernel_size,
        stride=config.group_size,
      )
    else:
      raise ValueError(f"Unknown pooling type: {config.pooling_type}")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: (Bsz, Ni, De)
    """
    # permute to (Bsz, De, Ni)
    x = x.permute(0, 2, 1)
    # left pad with (Ni % G) + Kc - 1 zeros
    pad_amount = (x.size(-1) % self.config.group_size) + self.config.conv_kernel_size - 1
    x = pad_left(x, pad_amount, pad_value=0.0)
    # conv layer: (Bsz, Dc, N)
    x = self.conv_layer(x)
    # activation: (Bsz, Dc, N)
    x = self.activation(x)
    # pooling: (Bsz, Dc, Ng) where Ng = ceil(Ni / G)
    x = left_padded_pooling(x, self.pooling)
    return x


class EncoderLayersOutputConfig(BaseConfig):
  """
  Output configuration using a number of encoder layers
  """

  num_layers: int
  encoder_block: EncoderBlockConfig


class OutputLayerConfig(UnionLikeConfig):
  """
  Configuration for the output layer.
  """

  mlp: Optional[MLPConfig] = None
  encoders: Optional[EncoderLayersOutputConfig] = None


class ConvPooledTransformerConfig(BaseConfig):
  """ """

  embedding_dims: int
  input_conv_pool: InputConvPoolConfig
  positional_encoding: PositionalEncodingConfig
  encoder_block: EncoderBlockConfig
  num_layers: int
  # This should be more complex since it needs
  # to predict multiple tokens at once
  output_layer: OutputLayerConfig


class ConvPooledTransformer(torch.nn.Module):
  def __init__(self, *, config: ConvPooledTransformerConfig, vocab_size: int):
    super().__init__()
    self.config = config
    self.vocab_size = vocab_size

    # embedding and positional encoding
    # then input convolution and pooling
    # then transformer encoder blocks
    # then group-wise output predictions layers
    self.char_embedding = torch.nn.Embedding(
      num_embeddings=vocab_size,
      embedding_dim=config.embedding_dims,
    )
    self.positional_encoding = PositionalEncoding(
      config=config.positional_encoding,
    )
    self.input_conv_pool = InputConvPool(
      config=config.input_conv_pool,
      emb_dims=config.embedding_dims,
    )
    self.transformer_dims = config.input_conv_pool.conv_out_channels
    self.encoder_blocks = torch.nn.ModuleList(
      [EncoderBlock(config.encoder_block, self.transformer_dims) for _ in range(config.num_layers)]
    )

    if config.output_layer.mlp is not None:
      self.output_layer = torch.nn.Sequential(
        MLP(config.output_layer.mlp, self.transformer_dims),
        torch.nn.Linear(
          config.output_layer.mlp.layer_dims[-1], vocab_size * config.input_conv_pool.group_size
        ),
      )
      self.output_type = "mlp"
    elif config.output_layer.encoders is not None:
      # separate output head for each group element prediction
      self.head_encoders = torch.nn.ModuleList(
        [
          torch.nn.ModuleList(
            [
              EncoderBlock(config.output_layer.encoders.encoder_block, self.transformer_dims)
              for _ in range(config.output_layer.encoders.num_layers)
            ]
          )
          for _ in range(config.input_conv_pool.group_size)
        ]
      )
      self.output_linears = torch.nn.ModuleList(
        [
          torch.nn.Linear(self.transformer_dims, vocab_size)
          for _ in range(config.input_conv_pool.group_size)
        ]
      )
      self.output_type = "encoders"
    else:
      raise ValueError("Unsupported output layer configuration")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: (Bsz, Ni)
    """
    # char embedding: (Bsz, Ni, De)
    x = self.char_embedding(x)
    # input convolution and pooling: (Bsz, Dc, Ng)
    x = self.input_conv_pool(x)
    # transformer encoder blocks
    # inputs must be (Bsz, Ng, Dc)
    x = x.permute(0, 2, 1)
    x = self.positional_encoding(x)
    attn_mask = torch.triu(
      torch.ones(x.size(1), x.size(1), device=x.device) * float("-inf"), diagonal=1
    )
    for encoder_block in self.encoder_blocks:
      x = encoder_block(x, is_causal=False, attn_mask=attn_mask)

    if self.output_type == "mlp":
      # group-wise output predictions: (Bsz, Ng, G * V)
      x = self.output_layer(x)
      # reshape to (Bsz, Ng * G, V) and return
      x = x.view(x.size(0), -1, self.vocab_size)

    elif self.output_type == "encoders":
      # separate output head for each group element prediction
      head_outputs = []
      for head in range(self.config.input_conv_pool.group_size):
        # head_input: (Bsz, Ng, Dc)
        head_input = x
        for encoder_block in self.head_encoders[head]:
          head_input = encoder_block(head_input, is_causal=False, attn_mask=attn_mask)
        head_output = self.output_linears[head](head_input)
        head_outputs.append(head_output)
      x = torch.cat(head_outputs, dim=-1)
      x = x.view(x.size(0), -1, self.vocab_size)

    return x


class ConvPooledTransformerWrapper(torch.nn.Module):
  def __init__(self, *, config: ConvPooledTransformerConfig, vocab_size: int):
    super().__init__()
    self.config = config
    self.vocab_size = vocab_size
    self.model = ConvPooledTransformer(config=config, vocab_size=vocab_size)

  def forward(self, batch: DictionaryBatch) -> DictionaryBatch:
    result = DictionaryBatch()
    x = batch["inputs"]
    result["logits"] = self.model(x)
    return result
