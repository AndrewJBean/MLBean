# MLBean
My ML Experimental Stuff

## Python Environment

The environment is managed with [uv](https://docs.astral.sh/uv/).

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`)
1. From the repo root, create the virtual environment and install everything (including dev tools):
    ```
    uv sync
    ```
    uv installs the Python version in `.python-version` if needed, and installs this repo in
    editable mode, so `MLBean` imports work without setting `PYTHONPATH`.
1. Run commands inside the environment with `uv run`, for example:
    ```
    uv run pytest
    uv run python -m MLBean.projects.transformers.train --dir=/path/to/checkpoints
    ```
    Alternatively, activate it with `source .venv/bin/activate`.
1. Add or remove dependencies with `uv add <package>` / `uv remove <package>` (use `--dev` for
   dev-only tools). Commit the updated `pyproject.toml` and `uv.lock`.

## Data

Pre-tokenize the training data into binary token shards:
```
uv run python -m MLBean.util.pretokenize --tokenizer=gpt2
```
Shards go to `~/prepared_data/<tokenizer>/` by default (override with `--out`). Stop anytime and
train on the shards already written; run again to resume from the parquet file/row cursor and
append more. Use `--max_tokens` for an absolute token budget, and `--overwrite` to start over.
