# madlabs/main-nixl: Andrew's dev line

madlabs/main carries what production serves (mad-llm-1: PLE table in RAM, no
NIXL tier). This branch is madlabs/main plus the pieces only Andrew's dev
machine uses:

- **The NVMe PLE reader**, `tools/ple_nvme/` (`0.2.0+systemone4`, its source
  guard pinned to madlabs/main's engine files), restored by `717d7f76e`
  (a revert of `861208ba9`, which removed it from madlabs/main), with its CI
  job and `.gitignore` lines.
- **Pennyroyal's two NIXL bounce-buffer fixes** by John Pezzulli: `caeb77668`
  (port of `067c639c0a`, the same patch as upstream #36524) and `d7d235814`
  (port of `e1a1572743`).

Only Andrew deploys this branch, and he qualifies it on his machine. It is
not production (owner, 2026-10-28).

## Keeping it in sync

It stays a superset of madlabs/main. Whoever moves madlabs/main merges it into
this branch the same day (`git merge origin/madlabs/main`, a merge commit, no
rebase), and Andrew may do the same whenever he needs. A merge never touches
the reader's files, because madlabs/main no longer has them. If a merge
changes one of the 11 engine modules the reader guards,
`python3 tools/ple_nvme/refresh_source_guard.py --check` fails: review the
reader against the change, re-pin with `--source madlabs/main@<sha>`, and bump
its `+systemone` version (`tools/ple_nvme/README.md`, "Source guard").

## Open items

1. **The NVMe PLE reader is re-pinned to this tree but has not run on a
   GPU.** Its guard pinned `madlabs/main@84f09ea980`, and 7 of its 11 modules
   differed here (`qsa.sparse_attn`, `qwen_sparse_attn_backend`, `qwen4_exp`,
   `qwen3_5`, `decode_cuda_graph_runner`, `server_args`, `utils.common`), so
   NVMe mode exited at startup. `tools/ple_nvme/REVIEW-2026-10-27.md` reviews
   the reader against those 7 changes. The guard now pins
   `madlabs/main@ae7ec4de75`, and the reader is `0.2.0+systemone4`.
   - The review found one break. This tree captures Qwen4-Exp prefill in
     breakable CUDA graphs by default, and on those replays the reader would
     add zero PLE rows without an error. In NVMe mode the reader now turns
     prefill graphs off, and refuses to start if the launch asks for one
     (`edcaf4109c`). Its PLE prefetch also raises inside any breakable CUDA
     graph, so a breakable graph selected some other way stops the server at
     startup. NVMe-mode prefill therefore runs eager, as Qwen4-Exp prefill
     does in a default launch on `madlabs/systemone-prod`, where Qwen4-Exp is
     left off the breakable-prefill list (`configs/model_config.py`:2312
     there).
   - Still to do on Andrew's box: reinstall the reader into a fresh
     `PENNY_PLE_PLUGIN_DIR`, key NIXL on `0.2.0+systemone4` if his launcher
     keys it on the reader version, run `ssd_stream/tests/gpu_smoke.py`, and
     serve once in NVMe mode, checking the tokens against a RAM-mode run with
     prefill graphs off too, greedy decoding and the same engine commit. RAM
     mode captures prefill graphs by default, so without that the comparison
     cannot tell the reader from graph-versus-eager numerics. The smoke test
     does not touch the model adapter, so the serve comparison is the only
     check of it. The steps and what the startup log should show are in
     `tools/ple_nvme/REVIEW-2026-10-27.md`, "Needs a GPU".
   - Compare `ssd_stream/Cargo.lock` with the crate versions in Andrew's
     built reader. The `+systemone4` bump changed only the reader's own
     version in the lock. The committed lock is the SSD
     Stream v0.2.0 release lock (`3c8fe2e97b`), not a record of his build.
     Until that commit the repo had no lock for the reader (`.gitignore`:232
     on madlabs/main ignores `Cargo.lock`), so a build's crate versions were
     whatever cargo resolved on that box when it ran.
     `3c8fe2e97b`'s message says the crate set `cargo generate-lockfile`
     picked "was never built with this code"; nothing records that.
   - Check the out-of-memory fix too. The `3a12b433f9` line changed the code
     the reader subclasses and calls (`qwen4.py:35` subclasses
     `Qwen4ExpPLELayer`). The review found no reader change needed for it,
     but nobody has run the decision-burst memory gate with the reader
     loaded.
   - mad-llm-deploy also has no launcher for this line in NVMe mode:
     `deploy/configs-trunk/serve-flash-next-madllm.sh:99-100` (main
     `ca92cfe3f0`) refuses anything but `PENNY_PLE_BACKEND=ram`.
2. **Pennyroyal's two NIXL bounce-buffer fixes are carried here, not on
   madlabs/main** (owner, 2026-10-28: NIXL runs only on Andrew's dev
   machine). They are `caeb776686` (port of `067c639c0a`, "bound NIXL hybrid
   bounce transfers", the same patch as upstream #36524, open) and
   `d7d2358142` (port of `e1a1572743`, "align mixed-state NIXL bounce I/O"),
   both authored by John Pezzulli. They were first committed as `72a1f20499`
   and `d42b47dfe2` on madlabs/nixl-bounce-fixes, which was deleted on
   2026-10-28; tag `archive/nixl-bounce-fixes` holds them.
3. **Andrew qualifies this branch on his dev machine** when he can,
   with the NIXL disk tier and the NVMe reader: item 1's GPU checks, and
   `test/registered/unit/mem_cache/test_hicache_nixl_storage.py` whole, in one
   process. Its tests now close their NIXL agents at teardown; on a Linux CPU
   container with the NIXL 1.4.0 wheel, the live agents left after the file
   fell from 4 to 0. Whether that also ends the crash seen on the GPU box is
   not yet known.
4. **mad-llm-deploy has an NVMe-mode serve config on a parked branch,**
   `feat/trunk-nvme-ple`, for the day Andrew's box is declared there. Its
   RAM launch is byte-identical to mad-llm-1's. The tag
   `archive/feat-trunk-nvme-ple` in mad-llm-deploy holds the branch's tip,
   `a3f13a3888`, and keeps the config if the branch is deleted.
