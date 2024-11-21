from absl import app, flags
from typing import Optional
import platform
import pathlib

from MLBean.configs.config_base import BaseConfig, UnionLikeConfig
from MLBean.data.dataset import DatasetConfig
from MLBean.modules.transformer_modules import (
  MLPConfig,
  RotaryAttentionConfig,
  RotaryEncoderBlockConfig,
  RotaryTransformerConfig,
  AutoregressiveTransformerConfig,
  SinusoidalPositionalEncodingConfig,
  PositionalEncodingConfig,
  MultiHeadAttentionConfig,
  MLPConfig,
  EncoderBlockConfig,
)
from MLBean.modules.iter_transformer import IterTransformerConfig
from MLBean.training.optimizer import OptimizerConfig, AdamWConfig
from MLBean.training.trainer import TrainerConfig, CheckpointingConfig, EvalConfig


DEFAULT_CONFIG_FILE = "all_config.json"

FLAGS = flags.FLAGS


def setup_flags():
  flags.DEFINE_string("dir", None, "The directory to load checkpoints from")
  flags.DEFINE_string("model", "rotary", "The model to use: rotary or iterated")
  flags.mark_flag_as_required("dir")


class ModelConfig(UnionLikeConfig):
  rotary: Optional[RotaryTransformerConfig] = None
  iterated: Optional[IterTransformerConfig] = None
  basic: Optional[AutoregressiveTransformerConfig] = None


class AllConfig(BaseConfig):
  model: ModelConfig
  training: TrainerConfig
  optimizer: OptimizerConfig
  dataset_train: DatasetConfig
  dataset_eval: Optional[DatasetConfig] = None


def get_basic_all_config(model: str = "rotary") -> AllConfig:
  if model == "rotary":
    return get_rotary_all_config()
  elif model == "iterated":
    return get_iterated_all_config()
  elif model == "basic":
    return get_basic_transformer_all_config()
  else:
    raise ValueError(f"Unknown model: {model}")


def get_rotary_all_config() -> AllConfig:
  emb_dims = 1024
  num_heads = 16
  num_layers = 12
  return AllConfig(
    model=ModelConfig(
      rotary=RotaryTransformerConfig(
        embedding_dims=emb_dims,
        encoder_block=RotaryEncoderBlockConfig(
          attention=RotaryAttentionConfig(
            num_heads=num_heads,
            min_theta=1.0,
            max_theta=1000.0,
          ),
          mlp=MLPConfig(layer_dims=[emb_dims, emb_dims]),
        ),
        num_layers=num_layers,
      )
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
      special_tokens_at_end=False,
    ),
  )


def get_iterated_all_config() -> AllConfig:
  emb_dims = 2048
  num_heads = 16
  input_layers = 3
  iter_layers = 4
  output_layers = 3
  num_iters = 5
  return AllConfig(
    model=ModelConfig(
      iterated=IterTransformerConfig(
        embedding_dims=emb_dims,
        encoder_block=RotaryEncoderBlockConfig(
          attention=RotaryAttentionConfig(
            num_heads=num_heads,
            min_theta=1.0,
            max_theta=10000.0,
          ),
          mlp=MLPConfig(layer_dims=[emb_dims, emb_dims]),
        ),
        input_layers=input_layers,
        iter_layers=iter_layers,
        output_layers=output_layers,
        num_iters=num_iters,
      )
    ),
    training=TrainerConfig(
      num_steps=1_000_000,
      log_interval=10,
      eval=EvalConfig(interval=500, steps=100),
      checkpointing=CheckpointingConfig(interval=1000) if platform.system() == "Darwin" else None,
    ),
    optimizer=OptimizerConfig(adamw=AdamWConfig(lr=0.00001)),
    dataset_train=DatasetConfig(
      batch_size=4,
      trunc_len=512,
      special_tokens_at_end=False,
    ),
  )


def get_basic_transformer_all_config() -> AllConfig:
  emb_dims = 1024
  num_heads = 16
  num_layers = 12
  return AllConfig(
    model=ModelConfig(
      basic=AutoregressiveTransformerConfig(
        embedding_dims=emb_dims,
        positional_encoding=PositionalEncodingConfig(
          sinusoidal=SinusoidalPositionalEncodingConfig(
            relative_freq_spacing=1.2,
            base_freq=1.0,
          ),
        ),
        encoder_block=EncoderBlockConfig(
          multi_head_attention=MultiHeadAttentionConfig(num_heads=num_heads),
          mlp=MLPConfig(layer_dims=[emb_dims, emb_dims]),
        ),
        num_layers=num_layers,
      ),
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


def get_all_config(
  chkpt_dir: pathlib.Path, maybe_create: bool = False, model: str = "rotary"
) -> AllConfig:
  all_config = get_basic_all_config(model)
  config_path = chkpt_dir / DEFAULT_CONFIG_FILE
  if (config_path).exists():
    all_config = AllConfig.json_load(config_path)
    print(f"succesfully loaded config from {config_path}")
    return all_config
  elif maybe_create:
    all_config.json_dump(config_path)
    print(f"Saved config to {config_path}")
    return all_config
  else:
    raise ValueError(f"Config file not found in {chkpt_dir}")


def main(argv):
  chkpt_dir = pathlib.Path(FLAGS.dir)
  _ = get_all_config(chkpt_dir, maybe_create=True, model=FLAGS.model)


if __name__ == "__main__":
  setup_flags()
  app.run(main)
