# Optional NVMe PLE for Flash-Next

Qwen3.8-Flash-Next carries a 47.68 GiB fp8 PLE (per-layer embedding) lookup
table. `--ple-offload-embedding` keeps it in pinned host RAM. This optional
reader instead serves the table from a file on local NVMe: each lookup reads
the exact rows through io_uring into two 16 MiB pinned staging buffers (plus
a 32 MiB reader pool) and copies them to the GPU. PLE precision and model
weights are unchanged. Page cache may still hold table pages, but the kernel
can reclaim them. Decode is slower than with the pinned table; use this when
host RAM, not speed, is the constraint.

The reader is an SGLang general plugin (`sglang.srt.plugins` entry point
`ssd_stream`). It changes nothing unless `PENNY_PLE_BACKEND=nvme` is set, so
installing it cannot affect other models or RAM mode. Once that mode is
selected, any failure stops the server; there is no fallback to RAM.

## Prepare once

The preparer copies only the PLE table bytes out of a complete local
checkpoint into a new directory (`ple/layer-<n>.bin` plus `ssd-stream.json`).
It symlinks every other asset to the original snapshot, which must stay in
place and unchanged. Budget about 48 GiB of SSD space. The output directory
must not already exist.

```bash
CARGO_BUILD_JOBS=2 PYTHON="$PWD/.venv/bin/python" bash tools/ple_nvme/install.sh
.venv/bin/python tools/ple_nvme/prepare_ple_nvme.py \
  --source /path/to/original-checkpoint \
  --output /path/on/local-nvme/flash-next-ple
```

`install.sh` needs Rust/Cargo and uv. It builds the reader (a PyO3 extension
with an io_uring page reader) into `.ple-nvme/`, a separate import directory,
and does not touch the Python environment. Set `PENNY_PLE_PLUGIN_DIR` to pick
another directory. The installer will not overwrite an existing one. The
overlay format is engine-independent: an overlay made for Pennyroyal v2.5.0
works here unchanged.

## Launch

The launcher must do all of the following:

```bash
export PENNY_PLE_BACKEND=nvme SGLANG_PLUGINS=ssd_stream
export PYTHONPATH="$REPO_ROOT/.ple-nvme:$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
# CPU-only preflight: overlay <-> source identity, reader version, entry
# point, source guard. Prints the manifest SHA-256 on success.
PLE_MANIFEST_SHA="$(CUDA_VISIBLE_DEVICES='' "$PYTHON" \
  "$REPO_ROOT/tools/ple_nvme/check_ple_nvme.py" \
  --source /path/to/original-checkpoint --prepared /path/on/local-nvme/flash-next-ple)"
sglang serve --model-path /path/on/local-nvme/flash-next-ple ...  # without --ple-offload-embedding
```

When `--model-path` points at the prepared overlay, the reader's
`ServerArgs.from_cli_args` hook forces `ple_offload_embedding=False`. It then
SHA-256s the whole table (51.2 GB, one sequential read) before any worker
starts.

Never serve the overlay without the reader active. If `PENNY_PLE_BACKEND=nvme`
is missing, or `SGLANG_PLUGINS` omits `ssd_stream`, SGLang does not detect the
missing PLE weights. For a bf16 `--dtype` it pins an uninitialised 47.68 GiB
table and serves garbage PLE rows. The launch steps above prevent this in the
launcher, but nothing checks a manual launch.

Keep the table and the source snapshot immutable while serving. If the
NIXL/HiCache L3 namespace is keyed on the PLE backend, key it on the manifest
SHA as well.

## Source guard

The reader registers its hooks only when every SGLang module listed in
`ssd_stream/src/sglang_ssd_stream/pennyroyal-source.json` is byte-identical to
the copy the adapter was reviewed against. That copy is currently
`madlabs/systemone-prod@50b964c017` (the SM120 online-MXFP8 port on
`madlabs/systemone-online-mxfp8`). If a rebase changes one of those modules,
NVMe startup and `tests/test_source_guard.py` fail, and nothing runs
unguarded. To accept the new code:

1. Review `qwen4.py`, `graph.py`, `config.py`, `offload.py` and `backend.py`
   against the changed modules.
2. Run `python tools/ple_nvme/refresh_source_guard.py --source madlabs/systemone-prod@<sha>`.
3. Bump the `+systemoneN` version in `pyproject.toml`, `Cargo.toml`,
   `__init__.py`, `check_ple_nvme.py` and `tests/test_pennyroyal.py`.
4. Reinstall into a fresh `PENNY_PLE_PLUGIN_DIR`.

## Tests

```bash
PYTHONPATH="$PWD/.ple-nvme:$PWD/python" .venv/bin/python -m pytest \
  tools/ple_nvme/tests tools/ple_nvme/ssd_stream/tests
```

`ssd_stream/tests/gpu_smoke.py` needs a GPU and a prepared table.

## Source and limits

`ssd_stream/` is an attributed adaptation of Garner McCloud's
[SSD Stream v0.2.0](https://github.com/garnermccloud/sglang-ssd-stream/tree/176a522ef9d6dbb5056ae1f467fe49af0f1258a5)
(Apache-2.0; its license is retained). It came here from Pennyroyal v2.5.0
(jpezzulli/sglang-rtxpro6000, commits 97871a356, 435023d4a and 239312931).
The reader, gather and graph adapter are unchanged from Pennyroyal. This port
changes only the source guard, the package version, file locations and the
tests that needed Pennyroyal's launcher. The upstream CLI runtime installer
is not exposed, and no replacement QSA/MTP runtime is included.

Pennyroyal qualified only TP1 Flash-Next on Linux x86_64 with Python 3.12. TP2,
CPU expert offload, prefill CUDA graphs and other speculative modes are
unsupported. Qwen4-Exp keeps prefill CUDA graphs disabled by default, and the
reader stages rows only for decode/verify graph replay, so do not force
`--cuda-graph-backend-prefill` on.
