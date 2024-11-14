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
)
from MLBean.training.optimizer import OptimizerConfig, AdamWConfig
from MLBean.training.trainer import TrainerConfig, CheckpointingConfig, EvalConfig


DEFAULT_CONFIG_FILE = "all_config.json"

FLAGS = flags.FLAGS


def setup_flags():
  flags.DEFINE_string("dir", None, "The directory to load checkpoints from")
  flags.mark_flag_as_required("dir")


class ModelConfig(UnionLikeConfig):
  rotary: Optional[RotaryTransformerConfig] = None


class AllConfig(BaseConfig):
  model: ModelConfig
  training: TrainerConfig
  optimizer: OptimizerConfig
  dataset_train: DatasetConfig
  dataset_eval: Optional[DatasetConfig] = None


def get_basic_all_config() -> AllConfig:
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


def get_all_config(chkpt_dir: pathlib.Path, maybe_create: bool = False) -> AllConfig:
  all_config = get_basic_all_config()
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
  all_config = get_all_config(chkpt_dir, maybe_create=True)


if __name__ == "__main__":
  setup_flags()
  app.run(main)
