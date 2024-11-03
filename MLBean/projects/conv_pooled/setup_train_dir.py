import pathlib
from absl import app, flags
import os

from MLBean.projects.conv_pooled.all_config import AllConfig, get_basic_all_config


FLAGS = flags.FLAGS


def setup_flags():
  flags.DEFINE_string("jobdir", None, "The directory to load checkpoints from")
  flags.mark_flag_as_required("jobdir")


def main(argv):
  chkpt_dir = pathlib.Path(FLAGS.jobdir)
  chkpt_dir.mkdir(parents=True, exist_ok=True)
  os.chdir(chkpt_dir)

  _ = get_all_config(chkpt_dir)


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


if __name__ == "__main__":
  setup_flags()
  app.run(main)
