from typing import Optional
import platform
import pathlib

from MLBean.configs.config_base import BaseConfig, UnionLikeConfig
from MLBean.data.dataset import DatasetConfig
from MLBean.modules.transformer_modules import (
  SinusoidalPositionalEncodingConfig,
  PositionalEncodingConfig,
  MultiHeadAttentionConfig,
  MLPConfig,
  EncoderBlockConfig,
)
from MLBean.modules.conv_pooled_transformer import (
  ConvPooledTransformerConfig,
  InputConvPoolConfig,
  OutputLayerConfig,
  EncoderLayersOutputConfig,
)
from MLBean.training.optimizer import OptimizerConfig, AdamWConfig
from MLBean.training.trainer import TrainerConfig, CheckpointingConfig, EvalConfig


class ConvModelConfig(UnionLikeConfig):
  conv_pooled_transformer: Optional[ConvPooledTransformerConfig] = None


class AllConfig(BaseConfig):
  model: ConvModelConfig
  training: TrainerConfig
  optimizer: OptimizerConfig
  dataset_train: DatasetConfig
  dataset_eval: Optional[DatasetConfig] = None


def get_basic_all_config() -> AllConfig:
  emb_dims = 256
  transformer_dims = 1024
  num_heads = 16
  num_layers = 8
  output_layers = 3
  group_size = 2
  conv_kernel_size = 6
  return AllConfig(
    model=ConvModelConfig(
      conv_pooled_transformer=ConvPooledTransformerConfig(
        embedding_dims=emb_dims,
        input_conv_pool=InputConvPoolConfig(
          group_size=group_size,
          conv_kernel_size=conv_kernel_size,
          conv_out_channels=transformer_dims,
          pooling_kernel_size=group_size,
        ),
        positional_encoding=PositionalEncodingConfig(
          sinusoidal=SinusoidalPositionalEncodingConfig(
            relative_freq_spacing=1.2,
            base_freq=1.0,
          ),
        ),
        encoder_block=EncoderBlockConfig(
          multi_head_attention=MultiHeadAttentionConfig(num_heads=num_heads),
          mlp=MLPConfig(layer_dims=[transformer_dims, transformer_dims]),
        ),
        num_layers=num_layers,
        output_layer=OutputLayerConfig(
          # mlp=MLPConfig(layer_dims=[2048] * 1, output_activation=True),
          encoders=EncoderLayersOutputConfig(
            num_layers=output_layers,
            encoder_block=EncoderBlockConfig(
              multi_head_attention=MultiHeadAttentionConfig(num_heads=num_heads),
              mlp=MLPConfig(layer_dims=[transformer_dims, transformer_dims]),
            ),
          ),
        ),
      ),
    ),
    training=TrainerConfig(
      num_steps=1_000_000,
      log_interval=100,
      eval=EvalConfig(interval=500, steps=100),
      checkpointing=CheckpointingConfig(interval=1000) if platform.system() == "Darwin" else None,
    ),
    optimizer=OptimizerConfig(adamw=AdamWConfig(lr=0.00001)),
    dataset_train=DatasetConfig(
      batch_size=4,
      trunc_len=512,
      group_size=group_size,
      special_tokens_at_end=False,
    ),
  )


def get_all_config(chkpt_dir: pathlib.Path) -> AllConfig:
  all_config = get_basic_all_config()
  if (chkpt_dir / "all_config.json").exists():
    all_config = AllConfig.json_load(chkpt_dir / "all_config.json")
    print(f"succesfully loaded config from {chkpt_dir / 'all_config.json'}")
    return all_config
  else:
    all_config.json_dump(chkpt_dir / "all_config.json")
    print(f"Saved config to {chkpt_dir / 'all_config.json'}")
    return all_config
