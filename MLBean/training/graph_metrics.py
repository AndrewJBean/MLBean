from absl import app, flags
import json
from matplotlib import pyplot as plt
import math

from collections import defaultdict

FLAGS = flags.FLAGS
flags.DEFINE_string(
  "f", None, "file path to the json lines file with logged metric values to graph"
)
flags.DEFINE_string("split", "eval", "train or eval split, default is eval")


def main(argv):
  with open(FLAGS.f, "r") as f:
    data = [json.loads(line) for line in f]

  # every line has a "step" key and other keys for metrics
  # Dict[metric_name, Dict[step, value]]
  metrics = defaultdict(lambda: ([], []))
  for d in data:
    if d["split"] != FLAGS.split:
      continue
    m = d["metrics"]
    for k, v in m.items():
      if k == "step":
        continue
      metrics[k][0].append(m["step"])
      metrics[k][1].append(v)

  num_graphs = len(metrics)
  num_cols = 1 if num_graphs < 3 else 2
  num_rows = int(math.ceil(num_graphs / num_cols))
  fig, axs = plt.subplots(num_rows, num_cols)
  if num_graphs == 1:
    axs = [[axs]]
  elif num_cols == 1:
    axs = [[a] for a in axs]
    print(len(axs))
    print(len(axs[0]))
    print(len(metrics))
  for i, (metric_name, metric_values) in enumerate(metrics.items()):
    x_loc = i % num_cols
    y_loc = i // num_cols
    axs[y_loc][x_loc].plot(metric_values[0], metric_values[1])
    axs[y_loc][x_loc].set_title(metric_name)
    axs[y_loc][x_loc].set_xlabel("step")
    axs[y_loc][x_loc].set_ylabel(metric_name)
  plt.show()


if __name__ == "__main__":
  app.run(main)
