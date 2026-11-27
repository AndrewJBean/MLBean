"""
Pre-tokenize a Hugging Face text dataset into flat binary token shards.

Each shard is a raw little-endian dump of token ids (`<u2` or `<u4`, chosen by vocab size).
No header. metadata.json is rewritten whenever a shard is completed, so a reader can use
whatever is already on disk. Call read_shard with the dtype from metadata.json; passing the
wrong dtype silently misinterprets the bytes.

Every document is preceded by the tokenizer's end-of-text token, which acts as a document
delimiter. Documents are packed back to back and may span shard boundaries.

Shards, metadata.json, progress.json, and buffer.bin are written directly to --out (default
~/prepared_data/<tokenizer>/). Stop whenever you like and train on the shards present; run
again to resume from the parquet file/row cursor and append more. Pass --overwrite to discard
the directory and start over.

Example:
  uv run python -m MLBean.util.pretokenize --tokenizer=gpt2
"""

import json
import logging
import os
import pathlib
import shutil
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from absl import app, flags
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.hf_file_system import HfFileSystem
from tqdm import tqdm

FLAGS = flags.FLAGS

PROGRESS_FILE = "progress.json"
BUFFER_FILE = "buffer.bin"
METADATA_FILE = "metadata.json"

TOKENIZERS: Dict[str, Callable[[], tiktoken.Encoding]] = {
  "gpt2": lambda: tiktoken.get_encoding("gpt2"),
}


def token_dtype(vocab_size: int) -> np.dtype:
  return np.dtype("<u2") if vocab_size <= 2**16 else np.dtype("<u4")


def default_out_dir(tokenizer: str) -> pathlib.Path:
  return pathlib.Path("~/prepared_data").expanduser() / tokenizer


def tokenize_docs(encoder: tiktoken.Encoding, texts: List[str], num_threads: int) -> np.ndarray:
  """Tokenizes texts into one flat array, with the end-of-text token before each document."""
  dtype = token_dtype(encoder.n_vocab)
  docs = encoder.encode_ordinary_batch(texts, num_threads=num_threads)
  out = np.empty(sum(len(d) for d in docs) + len(docs), dtype=dtype)
  pos = 0
  for doc in docs:
    out[pos] = encoder.eot_token
    out[pos + 1 : pos + 1 + len(doc)] = doc
    pos += 1 + len(doc)
  return out


def write_shard(path: pathlib.Path, tokens: np.ndarray):
  tmp = path.with_suffix(path.suffix + ".tmp")
  tokens.tofile(tmp)
  tmp.replace(path)


def read_shard(path: pathlib.Path, dtype: Union[np.dtype, str]) -> np.ndarray:
  """Memory-maps a shard. dtype must match metadata.json (e.g. "<u2"); it is not verified."""
  return np.memmap(path, dtype=np.dtype(dtype), mode="r")


def shard_name(index: int) -> str:
  return f"shard_{index:06d}.bin"


def run_fingerprint(dtype: np.dtype) -> Dict[str, Any]:
  return {
    "tokenizer": FLAGS.tokenizer,
    "dtype": dtype.str,
    "dataset": FLAGS.dataset,
    "dataset_config": FLAGS.dataset_config,
    "shard_size": FLAGS.shard_size,
  }


def list_source_parquets(dataset: str, dataset_config: str) -> List[str]:
  prefix = f"data/{dataset_config}/"
  files = [
    path
    for path in list_repo_files(dataset, repo_type="dataset")
    if path.startswith(prefix) and path.endswith(".parquet")
  ]
  if not files:
    raise FileNotFoundError(f"No parquet files under {prefix} in {dataset}")
  return sorted(files)


def parquet_num_rows(dataset: str, repo_path: str) -> int:
  """Reads only the parquet footer (via range request / cache), not the whole file."""
  fs = HfFileSystem()
  with fs.open(f"datasets/{dataset}/{repo_path}", "rb") as f:
    return pq.read_metadata(f).num_rows


def seek_from_docs_seen(dataset: str, files: List[str], docs_seen: int) -> Tuple[int, int]:
  """Maps a document count to (parquet_index, parquet_row) using per-file row counts."""
  remaining = docs_seen
  for index, repo_path in enumerate(files):
    num_rows = parquet_num_rows(dataset, repo_path)
    if remaining >= num_rows:
      remaining -= num_rows
      continue
    return index, remaining
  return len(files), 0


def ensure_local_parquet(dataset: str, repo_path: str) -> str:
  return hf_hub_download(dataset, repo_path, repo_type="dataset")


def iter_text_batches(
  dataset: str,
  files: List[str],
  *,
  start_file: int,
  start_row: int,
  batch_size: int,
) -> Iterator[Tuple[int, int, List[str]]]:
  """Yields (next_file_index, next_row, texts). next_* is the resume cursor after the batch."""
  for file_idx in range(start_file, len(files)):
    path = ensure_local_parquet(dataset, files[file_idx])
    print(f"Reading {files[file_idx]}", flush=True)
    skip = start_row if file_idx == start_file else 0
    row = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["text"]):
      texts = batch.column(0).to_pylist()
      if skip:
        if skip >= len(texts):
          skip -= len(texts)
          row += len(texts)
          continue
        texts = texts[skip:]
        row += skip
        skip = 0
      if not texts:
        continue
      row += len(texts)
      yield file_idx, row, texts
    start_row = 0


def validate_shard_files(out_dir: pathlib.Path, shards: List[Dict], dtype: np.dtype):
  for shard in shards:
    path = out_dir / shard["file"]
    if not path.exists():
      raise FileNotFoundError(f"Missing shard {path}")
    expected_bytes = shard["num_tokens"] * dtype.itemsize
    if path.stat().st_size != expected_bytes:
      raise ValueError(
        f"{path} has size {path.stat().st_size}, expected {expected_bytes} for "
        f"{shard['num_tokens']} tokens of {dtype.str}"
      )


def write_metadata(
  out_dir: pathlib.Path,
  *,
  encoder: tiktoken.Encoding,
  dtype: np.dtype,
  total_tokens: int,
  shards: List[Dict],
):
  metadata = {
    "tokenizer": FLAGS.tokenizer,
    "vocab_size": encoder.n_vocab,
    "eot_token": encoder.eot_token,
    "dtype": dtype.str,
    "dataset": FLAGS.dataset,
    "dataset_config": FLAGS.dataset_config,
    "total_tokens": total_tokens,
    "shards": shards,
  }
  tmp = out_dir / (METADATA_FILE + ".tmp")
  with open(tmp, "w") as f:
    json.dump(metadata, f, indent=2)
  tmp.replace(out_dir / METADATA_FILE)


def save_progress(
  out_dir: pathlib.Path,
  *,
  docs_seen: int,
  parquet_index: int,
  parquet_row: int,
  total_tokens: int,
  shards: List[Dict],
  buffer: np.ndarray,
  fill: int,
  fingerprint: Dict[str, Any],
):
  (out_dir / BUFFER_FILE).write_bytes(buffer[:fill].tobytes())
  payload = {
    **fingerprint,
    "docs_seen": docs_seen,
    "parquet_index": parquet_index,
    "parquet_row": parquet_row,
    "total_tokens": total_tokens,
    "fill": fill,
    "shards": shards,
  }
  tmp = out_dir / (PROGRESS_FILE + ".tmp")
  with open(tmp, "w") as f:
    json.dump(payload, f, indent=2)
  tmp.replace(out_dir / PROGRESS_FILE)


def load_progress(
  out_dir: pathlib.Path,
  fingerprint: Dict[str, Any],
  dtype: np.dtype,
  shard_size: int,
  files: List[str],
) -> Optional[Tuple[int, int, int, int, List[Dict], np.ndarray, int]]:
  """Returns docs_seen, parquet_index, parquet_row, total_tokens, shards, buffer, fill."""
  progress_path = out_dir / PROGRESS_FILE
  if not progress_path.exists():
    return None
  with open(progress_path) as f:
    progress = json.load(f)
  for key, value in fingerprint.items():
    if progress.get(key) != value:
      raise ValueError(
        f"{out_dir} was started with {key}={progress.get(key)!r}, "
        f"but this run has {key}={value!r}. Pass --overwrite to discard it."
      )
  shards = progress["shards"]
  validate_shard_files(out_dir, shards, dtype)
  fill = int(progress["fill"])
  buffer = np.empty(shard_size, dtype=dtype)
  if fill:
    loaded = np.fromfile(out_dir / BUFFER_FILE, dtype=dtype)
    if len(loaded) != fill:
      raise ValueError(
        f"{out_dir / BUFFER_FILE} has {len(loaded)} tokens, progress says fill={fill}"
      )
    buffer[:fill] = loaded
  total_tokens = int(progress["total_tokens"])
  expected_tokens = sum(s["num_tokens"] for s in shards) + fill
  if total_tokens != expected_tokens:
    raise ValueError(f"progress total_tokens={total_tokens} != shards+buffer ({expected_tokens})")
  docs_seen = int(progress["docs_seen"])
  if "parquet_index" in progress and "parquet_row" in progress:
    parquet_index = int(progress["parquet_index"])
    parquet_row = int(progress["parquet_row"])
  else:
    print("Progress has no parquet cursor; deriving it from docs_seen via file footers", flush=True)
    parquet_index, parquet_row = seek_from_docs_seen(FLAGS.dataset, files, docs_seen)
  return docs_seen, parquet_index, parquet_row, total_tokens, shards, buffer, fill


def clear_out_dir(out_dir: pathlib.Path):
  if out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True)


def prepare_out_dir(
  out_dir: pathlib.Path,
  overwrite: bool,
  fingerprint: Dict[str, Any],
  dtype: np.dtype,
  files: List[str],
) -> Tuple[int, int, int, int, List[Dict], np.ndarray, int]:
  """Returns docs_seen, parquet_index, parquet_row, total_tokens, shards, buffer, fill."""
  if overwrite:
    clear_out_dir(out_dir)
  else:
    out_dir.mkdir(parents=True, exist_ok=True)

  resumed = load_progress(out_dir, fingerprint, dtype, FLAGS.shard_size, files)
  if resumed is not None:
    docs_seen, parquet_index, parquet_row, total_tokens, shards, buffer, fill = resumed
    print(
      f"Resuming {out_dir}: {len(shards)} shards, {total_tokens:,} tokens, "
      f"cursor file {parquet_index}:{parquet_row} ({docs_seen:,} docs)",
      flush=True,
    )
    return docs_seen, parquet_index, parquet_row, total_tokens, shards, buffer, fill

  leftovers = [
    p
    for p in out_dir.iterdir()
    if p.name.endswith(".bin") or p.name in {METADATA_FILE, PROGRESS_FILE, BUFFER_FILE}
  ]
  if leftovers:
    raise FileExistsError(
      f"{out_dir} has data but no usable {PROGRESS_FILE}; pass --overwrite to discard it"
    )
  buffer = np.empty(FLAGS.shard_size, dtype=dtype)
  return 0, 0, 0, 0, [], buffer, 0


def setup_flags():
  flags.DEFINE_enum("tokenizer", "gpt2", list(TOKENIZERS), "The tokenizer to use")
  flags.DEFINE_string("out", None, "Output directory. Defaults to ~/prepared_data/<tokenizer>/")
  flags.DEFINE_string("dataset", "HuggingFaceFW/fineweb-edu", "Hugging Face dataset name")
  flags.DEFINE_string("dataset_config", "CC-MAIN-2013-20", "Hugging Face dataset config")
  flags.DEFINE_integer("shard_size", 100_000_000, "Number of tokens per shard")
  flags.DEFINE_integer("max_tokens", None, "Stop after about this many tokens (default: all)")
  flags.DEFINE_integer("num_threads", os.cpu_count(), "Number of tokenizer threads")
  flags.DEFINE_integer("docs_per_batch", 1024, "Number of documents tokenized per batch")
  flags.DEFINE_bool(
    "prefetch_parquets",
    False,
    "Download all source parquet files before tokenizing",
  )
  flags.DEFINE_bool(
    "overwrite",
    False,
    "Delete the output directory and start over",
  )


def main(argv):
  for name in ["httpx", "httpx2"]:
    logging.getLogger(name).setLevel(logging.WARNING)
  out_dir = pathlib.Path(FLAGS.out).expanduser() if FLAGS.out else default_out_dir(FLAGS.tokenizer)

  encoder = TOKENIZERS[FLAGS.tokenizer]()
  dtype = token_dtype(encoder.n_vocab)
  fingerprint = run_fingerprint(dtype)
  files = list_source_parquets(FLAGS.dataset, FLAGS.dataset_config)
  print(f"Found {len(files)} parquet files in {FLAGS.dataset}/{FLAGS.dataset_config}", flush=True)

  docs_seen, parquet_index, parquet_row, total_tokens, shards, buffer, fill = prepare_out_dir(
    out_dir, FLAGS.overwrite, fingerprint, dtype, files
  )
  print(f"Tokenizer {FLAGS.tokenizer}: vocab size {encoder.n_vocab}, storing as {dtype.str}")
  print(f"Writing shards to {out_dir}", flush=True)

  if FLAGS.prefetch_parquets:
    for repo_path in files:
      ensure_local_parquet(FLAGS.dataset, repo_path)

  def flush():
    nonlocal fill
    name = shard_name(len(shards))
    write_shard(out_dir / name, buffer[:fill])
    shards.append({"file": name, "num_tokens": fill})
    fill = 0
    write_metadata(
      out_dir,
      encoder=encoder,
      dtype=dtype,
      total_tokens=sum(s["num_tokens"] for s in shards),
      shards=shards,
    )

  target = FLAGS.max_tokens
  # --max_tokens is an absolute token budget for this directory (including prior shards).
  shards_before = len(shards)
  stopped_for_budget = False
  progress = tqdm(
    total=target,
    initial=total_tokens,
    unit="tok",
    unit_scale=True,
    mininterval=5.0,
  )
  for parquet_index, parquet_row, texts in iter_text_batches(
    FLAGS.dataset,
    files,
    start_file=parquet_index,
    start_row=parquet_row,
    batch_size=FLAGS.docs_per_batch,
  ):
    if target is not None and total_tokens >= target:
      stopped_for_budget = True
      break
    tokens = tokenize_docs(encoder, texts, FLAGS.num_threads)
    while len(tokens):
      if target is not None:
        n = min(FLAGS.shard_size - fill, len(tokens), target - total_tokens)
      else:
        n = min(FLAGS.shard_size - fill, len(tokens))
      if n <= 0:
        break
      buffer[fill : fill + n] = tokens[:n]
      fill += n
      total_tokens += n
      tokens = tokens[n:]
      if fill == FLAGS.shard_size:
        flush()
    docs_seen += len(texts)
    progress.update(total_tokens - progress.n)
    save_progress(
      out_dir,
      docs_seen=docs_seen,
      parquet_index=parquet_index,
      parquet_row=parquet_row,
      total_tokens=total_tokens,
      shards=shards,
      buffer=buffer,
      fill=fill,
      fingerprint=fingerprint,
    )
    if len(shards) > shards_before:
      print(
        f"shard {len(shards) - 1}: {sum(s['num_tokens'] for s in shards):,} tokens on disk, "
        f"cursor {parquet_index}:{parquet_row}",
        flush=True,
      )
      shards_before = len(shards)
    if target is not None and total_tokens >= target:
      stopped_for_budget = True
      break
  progress.close()

  # Keep a partial shard in buffer.bin when stopping early so the next run can keep packing.
  # Flush a short final shard only when the source is exhausted.
  if fill > 0 and not stopped_for_budget:
    flush()
    save_progress(
      out_dir,
      docs_seen=docs_seen,
      parquet_index=parquet_index,
      parquet_row=parquet_row,
      total_tokens=total_tokens,
      shards=shards,
      buffer=buffer,
      fill=0,
      fingerprint=fingerprint,
    )
  elif shards:
    write_metadata(
      out_dir,
      encoder=encoder,
      dtype=dtype,
      total_tokens=sum(s["num_tokens"] for s in shards),
      shards=shards,
    )

  on_disk = sum(s["num_tokens"] for s in shards)
  print(
    f"{on_disk:,} tokens in {len(shards)} shards at {out_dir}"
    + (f" (+{fill:,} tokens buffered)" if fill else ""),
    flush=True,
  )


if __name__ == "__main__":
  setup_flags()
  app.run(main)
