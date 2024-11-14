from typing import Optional
import platform
import pathlib

from MLBean.configs.config_base import BaseConfig
from MLBean.data.dataset import DatasetConfig
from MLBean.modules.transformer_modules import (
  AutoregressiveTransformerConfig,
  SinusoidalPositionalEncodingConfig,
  PositionalEncodingConfig,
  MultiHeadAttentionConfig,
  MLPConfig,
  EncoderBlockConfig,
)
from MLBean.training.optimizer import OptimizerConfig, AdamWConfig
from MLBean.training.trainer import TrainerConfig, CheckpointingConfig, EvalConfig


class AllConfig(BaseConfig):
  model: AutoregressiveTransformerConfig
  training: TrainerConfig
  optimizer: OptimizerConfig
  dataset_train: DatasetConfig
  dataset_eval: Optional[DatasetConfig] = None


def get_basic_all_config() -> AllConfig:
  return AllConfig(
    model=AutoregressiveTransformerConfig(
      embedding_dims=1024,
      positional_encoding=PositionalEncodingConfig(
        sinusoidal=SinusoidalPositionalEncodingConfig(
          relative_freq_spacing=1.2,
          base_freq=1.0,
        ),
      ),
      encoder_block=EncoderBlockConfig(
        multi_head_attention=MultiHeadAttentionConfig(num_heads=16),
        mlp=MLPConfig(layer_dims=[1024, 1024]),
      ),
      num_layers=16,
    ),
    training=TrainerConfig(
      num_steps=100000,
      log_interval=100,
      eval=EvalConfig(interval=500, steps=100),
      checkpointing=CheckpointingConfig(interval=1000) if platform.system() == "Darwin" else None,
    ),
    optimizer=OptimizerConfig(adamw=AdamWConfig(lr=0.00005)),
    dataset_train=DatasetConfig(batch_size=4, trunc_len=512),
  )


def get_all_config(chkpt_dir: pathlib.Path) -> AllConfig:
  all_config = get_basic_all_config()
  if (chkpt_dir / "all_config.json").exists():
    all_config = AllConfig.json_load(chkpt_dir / "all_config.json")
    print(f"succesfully loaded config from {chkpt_dir / 'all_config.json'}")
    return all_config
  elif (chkpt_dir / "config.json").exists():
    all_config.model = AutoregressiveTransformerConfig.json_load(chkpt_dir / "config.json")
    all_config.json_dump(chkpt_dir / "all_config.json")
    print("Loaded model config from", chkpt_dir / "config.json")
    print(f"Saved config to {chkpt_dir / 'all_config.json'}")
    return all_config
  else:
    all_config.json_dump(chkpt_dir / "all_config.json")
    print(f"Saved config to {chkpt_dir / 'all_config.json'}")
    return all_config
