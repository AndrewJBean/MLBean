from typing import Optional, List
import os
import math

import torch

from MLBean.configs.config_base import UnionLikeConfig, BaseConfig
from MLBean.modules.transformer_modules import (
  RotaryEncoderBlockConfig,
  RotaryEncoderBlock,
  MLPConfig,
  MLP,
)


class IterTransformerConfig(BaseConfig):
  """
  Configuration for the Iterative Transformer.

  Args:
    embedding_dims: The dimension of the input embeddings.
    encoder_block: The configuration for the encoder block.
    input_layers: The number of input layers.
    iter_layers: The number of iterative layers.
    output_layers: The number of output layers.
  """

  embedding_dims: int
  encoder_block: RotaryEncoderBlockConfig
  input_layers: int
  iter_layers: int
  output_layers: int
  num_iters: int


class IterTransformer(torch.nn.Module):
  """
  Iterative Transformer model.
  """

  def __init__(self, config: IterTransformerConfig, vocab_size: int):
    super().__init__()
    self.embedding = torch.nn.Embedding(vocab_size, config.embedding_dims)
    self.num_iters = config.num_iters
    self.input_encoder_blocks = torch.nn.Sequential(
      *[
        RotaryEncoderBlock(config.encoder_block, config.embedding_dims)
        for _ in range(config.input_layers)
      ]
    )
    self.iter_encoder_blocks = torch.nn.ModuleList(
      [
        RotaryEncoderBlock(config.encoder_block, config.embedding_dims)
        for _ in range(config.iter_layers)
      ]
    )
    self.output_encoder_blocks = torch.nn.Sequential(
      *[
        RotaryEncoderBlock(config.encoder_block, config.embedding_dims)
        for _ in range(config.output_layers)
      ]
    )
    self.output_projection = torch.nn.Linear(config.embedding_dims, vocab_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
    """
    Forward pass of the model.

    Args:
      x: The input tensor, shape (batch_size, num_tokens).

    Returns:
      The output tensor, shape (batch_size, num_tokens, vocab_size).
    """
    x = self.embedding(x)  # => (batch_size, num_tokens, embedding_dims)
    x = self.input_encoder_blocks(x)  # => (batch_size, num_tokens, embedding_dims)
    # len of iter_layer_states will be num_iters + 1
    iter_layer_states = [x]
    for _ in range(self.num_iters):
      for encoder_block in self.iter_encoder_blocks:
        x = encoder_block(x)
      iter_layer_states.append(x)

    # each output has shape (batch_size, num_tokens, vocab_size)
    if self.training:
      return [self.output_projection(state) for state in iter_layer_states]
    # just return the last if not training
    return self.output_projection(iter_layer_states[-1])
