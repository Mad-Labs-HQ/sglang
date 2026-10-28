# Mad Labs SGLang trunk: madlabs/main

Branch `madlabs/main` is upstream sgl-project/sglang at **`fe89a7c890`**
(2026-10-18) plus our commits. Our production box has served
**`922810a5ed`** (tag `prod/2026-10-27`) since 2026-10-27 06:44 UTC: 55
first-parent commits past the base, one of them a merge (`b1834f39fc`) that
brings in no code. The commits after it change the fork's docs, CI and tests,
and change and then remove the NVMe PLE reader (`tools/ple_nvme/`, now on
`madlabs/main-nixl`). None of them touches `python/` or `sgl-kernel/`:
`git diff --stat prod/2026-10-27 origin/madlabs/main -- python sgl-kernel`
prints nothing.

`madlabs/main` is the fork's default branch. The branch `main` is a plain copy
of upstream at `67bb6a58d0` (2026-09-25) and carries nothing of ours. Engines
are built from the branch names below, never from `main`.

Where a source below is in **mad-llm-deploy** (our private deployment repo),
the path is at commit `0c63afe`, the release record of `922810a5ed`, unless it
says `main`. "CHANGELOG" means mad-llm-deploy `deploy/CHANGELOG.md`, and
"engine doc" means mad-llm-deploy `docs/design/mad-llm-engine.md`.

## How the line got here

| Date | Event | Source |
|---|---|---|
| 2026-09-25 | Andrew Stanton's System One line on upstream `67bb6a58d0`, 29 commits up to `f9bfb5802d` | `git log 67bb6a58d0..f9bfb5802d` |
| 2026-10-02 | The owner makes that line the trunk and asks for it to move onto current upstream | engine doc:265-267 |
| 2026-10-18 | Rebased onto `fe89a7c890` as `f132f9b522`; 1,440 upstream commits newer than `67bb6a58d0` | `git rev-list --count 67bb6a58d0..fe89a7c890` |
| 2026-10-21 | The prefill out-of-memory fix, `142800d641` through `3a12b433f9`, committed to this line | `git log --format=%cd`; CHANGELOG:16-28 |
| 2026-10-26 | `b1834f39fc` merges the old madlabs/main (`f9bfb5802d`) without taking its code, so madlabs/main can move here by fast-forward; then `922810a5ed` | `git show b1834f39fc` |
| 2026-10-27 01:16 UTC | madlabs/main moves from `f9bfb5802d` to `922810a5ed` by fast-forward | the fork's activity log |
| 2026-10-27 | `922810a5ed` serves production | CHANGELOG:11-13, 47 |

## What we carry

Every first-parent commit since the base, by theme, oldest first. Commit
bodies give each defect, fix and port note. "Upstream" is the state read from
sgl-project/sglang on 2026-10-27.

| Theme | Commits | What and why | Upstream |
|---|---|---|---|
| System One decision routes (Andrew Stanton) | `19f9e3bd54`, `4da953a2f8`, `c393e6bb0e`, `a3373a8269`, `12b6a7e8f5`, `a857db7990`, `ce37ddda06`, `8aa0911254`, `822c95febb`, `0665e5b93f` | `/v1/systemone` and `/v1/decisions` (`http_server.py`); calibration, moved to the client in `ce37ddda06`; one prefill of the prefix a request's prompts share; benchmarks and results in `benchmark/systemone/` | none; product work |
| qwen3_coder tool calls | `c9fcb0afed`, `5c66c4e68b` (Andrew Stanton, ports of his `ed6e799b22`, `c32797570b` on madlabs/pennyroyal-v2.5.0); `a60f54920e`, `38ef1f4b15`, `e2fcbaa294` (John Pezzulli); `2524a3f19e` (their tests' lint) | unwraps a stray `arguments` envelope; streams string arguments as they are generated; keeps prose and separators between parallel calls; validates wrappers and tool names | #37408 and #38624 open; incremental streaming #21829 closed unmerged |
| Scheduler and sampling fixes | `1565178251` (Andrew Stanton, port of his `8dcd3ec476` on madlabs/pennyroyal-v2.5.0); `6861f13982`, `4758fbd8c7`, `c902e9e40d` (John Pezzulli) | no `.tolist()` on logprob values already converted (a crash); skip the token-ID collective on a one-rank group; trim speculative grammar history in place; the structured-output warmup | logprob crash not reported; #38489 closed unmerged 2026-10-14; #38865 open; warmup is ours only |
| Reasoning effort | `0e943e7da8` (John Pezzulli) | a request's own reasoning effort wins over the server default | #38338 open (issue #38104) |
| Prefix-cache opt-out | `2f46dde87c` (Andrew Stanton, port of his `2a7ca28d36` on madlabs/pennyroyal-v2.5.0); tests `f96e2b1ce0`, `5064aa1c67` | `skip_cache_insert`: one-shot scoring prompts are read from the cache but not published to it. Python front end only: the Rust front end does not send it (`io_struct.py:1196-1198`) | #38738 open |
| Mamba admission and host memory | `464ff30a0e` (Andrew Stanton, port of his `c9eb6c2e2d` on madlabs/pennyroyal-v2.5.0); `bb6d7c2e53` (Justin Huang); `452e0447be` (Andrew Stanton); `538f6e23b1`, `f828156f40`, `85c31de269`, `f132f9b522` (ours) | a full Mamba pool skips a chunk's checkpoint instead of asserting; atomic Mamba admission; a host-memory hit's restored state is charged; `--hicache-mamba-size` declares the host Mamba share (why: engine doc:197-202); a chunked-prefill checkpoint counts no cache hit, undoing a regression from #41520 (engine doc:204-210); keep upstream's side-state path beside ours | assert not reported; #39787 open; #41520 regression not reported |
| PLE slot state through host memory and NIXL | `9c425b89f1` (Andrew Stanton, from Pennyroyal `516e42a2ee` + `bcd51eeb0e`) | Flash-Next's PLE side state is saved and restored with its Mamba state, through host memory and the NIXL and Mooncake tiers | #39862 (merged 2026-10-04, in the base) covers host memory only; the NIXL part is needed by Andrew's disk tier (engine doc:158) |
| SM120 online MXFP8 | `95300400b0`, `ce34895c4e`, `20de35f49b`, `f802f591c7`, `6dae25d1d1` (Andrew Stanton, from Pennyroyal `97871a3566`); `5bba1e849b`, `e6098e4b87` (ours: test entry points, import order) | 8-bit weights for Flash-Next's BF16 projections, the hyperconnection mix and `lm_head` on RTX PRO 6000 (SM120). Switch `SGLANG_SM120_ONLINE_MXFP8`, default off (`environ.py:1106`) | none; specific to this card |
| Request lifecycle spans | `f723260b9d` (John Pezzulli, from Pennyroyal `730d244a08`) | per-request timing spans that our analysis reads | none |
| Image preprocessing device | `413719a8b2`, `8bd82e73d3` (John Pezzulli) | `SGLANG_MM_PREPROCESS_DEVICE` (`environ.py:1501`) keeps image decoding off the GPU in the front-end process, which otherwise holds a second ~630 MiB on the card (engine doc:227) | none |
| Prefill out-of-memory fix | `142800d641` (Qiaolin Yu); `8a18f5888f` (shi yang); `d2874d97af`, `01b10d5751`, `fe0d496131`, `3a12b433f9` (ours) | `f132f9b522` ran out of GPU memory under decision bursts with prefill CUDA graphs on (2026-10-20). Fix: the packed PLE convolution and freed per-bucket outputs, then hardening and bitwise, memory and fallback tests (CHANGELOG:26-28) | #44150 merged 2026-10-19 as `9ab74b8517`, after our base: drop `142800d641` at the next rebase. #43032 open |
| API-key redaction | `922810a5ed` | `--api-key` and `--admin-api-key` were printed in the startup log, `/server_info` and the recorded launch command | issue #37457 open |

Pennyroyal is John Pezzulli's public RTX PRO 6000 build of Flash-Next
(jpezzulli/sglang-rtxpro6000, tag `pennyroyal-v2.5.0`). Its fork branch
`madlabs/pennyroyal-v2.5.0`, Pennyroyal plus Andrew Stanton's five production
fixes, is our rollback line (engine doc:77-82). It shares no fork commits with
madlabs/main, so fixes move between the two by porting (engine doc:68-71).
What we chose not to carry from it, and why: engine doc:212-231.

## How madlabs/main moves

Only by fast-forward. Work lands on a branch off madlabs/main, is qualified,
and then madlabs/main is moved to it with a plain push, which git refuses
unless it is a fast-forward. Never force-push it. Since 2026-10-28 GitHub
enforces this: the fork's ruleset "madlabs production lines" refuses a
force-push to, or a deletion of, `madlabs/main`, `madlabs/main-nixl`,
`madlabs/pennyroyal-v2.5.0` and `madlabs/v0.5.19-prod`, and the ruleset
"release and archive tags" refuses deleting or moving a `prod/*` or
`archive/*` tag.

A rebase onto newer upstream rewrites every commit, so it cannot reach
madlabs/main by fast-forward. It lands the way the 2026-10-26 merge did
(pushed to madlabs/main 2026-10-27 01:16 UTC): the rebased line merges the old
madlabs/main with `-s ours`, which records the old tip as an ancestor and keeps
the rebased tree unchanged.

### The 2026-10-26 merge, `b1834f39fc`

Parents: `3a12b433f9` (the rebased line) and `f9bfb5802d` (the old
madlabs/main, also the tip of `madlabs/systemone-prod`). Its tree is
`3a12b433f9`'s tree (`e6be62479b`): the merge brings in no code.

Its message says the 29 old-line commits are "13 identical, 14 ported to the
newer upstream, and two replaced on purpose". Checked with `git cherry -v
3a12b433f9 f9bfb5802d 67bb6a58d0` and `git range-diff
67bb6a58d0..f9bfb5802d fe89a7c890..3a12b433f9`:

- **13 identical**: the same patch is on the rebased line.
- **14 ported**: 13 are paired by range-diff with an adapted patch. The 14th,
  `c7e476e148` (#39893), needed no port: upstream merged it as `89f21671bb`,
  which is in the base.
- **2 replaced on purpose**:
  - `d046751e28`, a flat 4-slot Mamba reservation, is retired by
    `bb6d7c2e53` plus `452e0447be` (that commit's message says so).
  - `c8333cb8fa`, the reader re-pin to `+systemone2`, was redone as
    `5a213706ea`: a cherry-pick re-pinned to `84f09ea980` and `+systemone3`
    for the `f6fcda8827` tree, not for this one (see the Correction below).

**Correction.** The merge message says `c8333cb8fa` was replaced by "the NVMe
PLE reader re-pinned for this tree as systemone3". That is false. `5a213706ea`
pinned the reader's guard to `madlabs/main@84f09ea980`, a commit of
madlabs/main-next on upstream `f6fcda8827` that is not in this tree's history
(`git merge-base --is-ancestor 84f09ea980 922810a5ed` fails). madlabs/main-next
was deleted on 2026-10-28; tag `archive/main-next-jpeg` holds its tip and
`84f09ea980`. Nobody re-pinned
it for `fe89a7c890` until 2026-10-27; at `922810a5ed`, 7 of the 11 modules it
guards differed (`tools/ple_nvme/REVIEW-2026-10-27.md` on `madlabs/main-nixl`). The
body of `5a213706ea` also still describes the `f6fcda8827` rebase and a
`+systemone2` deploy note. Published commits are not rewritten, so the
correction lives here.

Because of `-s ours`, git reports the 29 old-line commits as part of
madlabs/main (`git branch --contains`, `git branch --merged`, `git log
madlabs/main -- <path>`), but the tree holds only the ports. Answer "is commit
X in production" with `git cherry` or `git range-diff`, not with reachability.
Never pick branches to delete with `git branch --merged`: it lists
`madlabs/systemone-prod`, which is still Andrew's live branch.

## How to build

mad-llm-deploy `main` has the installer, `deploy/install-sglang-flash-next-trunk.sh`:

- `ENGINE_REF` (the branch) and `ENGINE_COMMIT` (the pin) come from the box's
  declaration and have no defaults (:32-33).
- It clones the fork by branch name and checks out the pin (:57-61), so a
  served commit must stay reachable from that branch.
- It builds with `deploy/build-pennyroyal.sh` (:103-108). That script is the
  one place the qualified build flags are written: the cu130 wheel index,
  `--no-build-isolation`, editable `python/` (`build-pennyroyal.sh`:65-72).
- When the declaration asks for the Rust tree core, it installs the toolchain
  that `rust/rust-toolchain.toml` pins (:111-138).

The GPU library stack is pinned in `python/pyproject.toml`: `torch==2.14.1`
(:6, :89), `flashinfer_python[cu13]==0.7.2` (:42), `sglang-kernel==0.4.10`
(:81).

The NVMe PLE reader is not on this line: it lives on `madlabs/main-nixl`
("Andrew's dev line" below).

## How it is qualified

`pod qualify` (mad-llm-deploy `pod/madpod/qualify/README.md`) boots the
candidate on a scratch box with the production card, runs a plan of
measurements, judges them against a baseline, and tears the box down. It
refuses to run on a production box (:7-9).

`922810a5ed` was qualified by run `20261026T180342Z` on our replica, with the
same commit and settings as production, from 2026-10-26 18:03 to 10-27 00:54
UTC (CHANGELOG:29-38):

- verify 12/12 on all 7 boots; GPU unit tests 2,296 passed, 0 failed;
- decision burst 498/498 with no out-of-memory, lowest free GPU memory
  2,252 MiB (the unfixed `f132f9b522`: 658-860 MiB);
- returning paused agents reused 96.87-96.88% of their cache on three boots
  (the predecessor: 3.4-5.0%).

The owner waived two gate limits on 2026-10-27, since every build of this line
fails them (CHANGELOG:40-43):

- short-prompt sameness against `c9eb6c2e2d`: 70 of 1,536 reference tokens
  differ, against a limit of 58;
- decision-burst p99: 251 s, against a limit of 180 s.

What that qualification did not cover:

- **Andrew's dev setup** (a NIXL disk tier and the NVMe PLE reader). It is not
  production and not on this line; see "Andrew's dev line".
- **Most of the fork's CPU tests.** The unit-test gate ran a hand-picked list
  (`pod/madpod/qualify/box/unit-tests.trunkmove.txt`, 25 files). It has 14 of
  the 34 CPU-registered test files the fork adds or changes, and none of the
  four upstream CPU tests our changes had broken.
- **The NIXL storage test file as a whole.** It crashed when its tests shared
  one process; each test passed alone (CHANGELOG:43-44).

### CPU tests

`.github/workflows/madlabs-cpu.yml` names the files and runs them on a hosted
runner:

- `cpu-unit` runs 38 files: the 34 CPU-registered files the fork adds or
  changes, and the four upstream CPU tests our changes had broken
  (`test_parity`, `test_prefill_result_polling`,
  `test_scheduler_chunked_req_gate`, `test_server_args_docs_coverage`).

GitHub Actions is turned off on this fork. Turning it on also arms upstream's
workflow files that fire on any pull request, `pr-test.yml` among them (its
`pull_request:` trigger has no branch filter). Its GPU stages run on upstream's
self-hosted runner labels (`scripts/ci/runner_configs.yml:21-27`). So turning
Actions on is an owner decision, and the upstream workflows must be turned off
with it. Until
then, run the same list locally before each move of madlabs/main:

```bash
uv pip install -e "python[dev]" --index-strategy unsafe-best-match --prerelease allow
for f in $(sed -n '/CPU_TEST_FILES: |/,/^    steps:/p' .github/workflows/madlabs-cpu.yml | grep -o 'test/registered/[^ ]*\.py'); do
  python -m pytest -q -p no:cacheprovider "$f" || echo "FAILED $f"
done
```

## Production engines

Every engine that has served production, from the CHANGELOG, and the tag
that keeps it. Each tag is named for the date its engine began serving and
points at that engine commit; tags cannot be deleted or moved, so an engine
stays fetchable whatever happens to its branch. Pushing tags needs the
owner's approval (engine doc:271-274).

| Served from | Engine | Tag | Branch | CHANGELOG |
|---|---|---|---|---|
| 2026-08-24 | `9a906687f0` (Qwen3.8-27B with DFlash2) | `prod/2026-08-24` | `madlabs/v0.5.19-prod` | :134-136 |
| 2026-09-22 | `8dcd3ec476` (Flash-Next on Pennyroyal) | `prod/2026-09-22` | `madlabs/pennyroyal-v2.5.0` | :100-102 |
| 2026-10-01 | `c9eb6c2e2d` (Pennyroyal; the rollback for `922810a5ed`) | `prod/2026-10-01` | `madlabs/pennyroyal-v2.5.0` | :57-59, :47 |
| 2026-10-27 | `922810a5ed` | `prod/2026-10-27` | `madlabs/main` | :11-13 |

The `archive/*` tags keep the tips of branches deleted on 2026-10-28 that no
remaining branch contains, each named after its branch:
`archive/host-tier-ports`, `archive/main-next-jpeg` (which also holds
madlabs/main-next's tip), `archive/nixl-bounce-fixes`,
`archive/spike/mamba-cache-trace`, `archive/spike/mamba-cache-trace-pennyroyal`,
`archive/systemone-ports` and `archive/trunk-ports`. The other branches
deleted that day had tips already on madlabs/main.

## One production engine

The rule: every production deployment runs the same engine line, so the
deployments never drift apart (owner decision, 2026-10-04; engine doc:23-27).

It is **met**: mad-llm-1 is the only production deployment, and it runs
`922810a5ed` (tag `prod/2026-10-27`) from madlabs/main. Andrew's box is a dev
machine, not a production deployment (owner, 2026-10-28): it is the only box
that runs a NIXL disk tier and the NVMe PLE reader.

## Andrew's dev line: `madlabs/main-nixl`

This line carries only what production serves. Andrew's dev-only pieces live
on **`madlabs/main-nixl`** = madlabs/main + the NVMe PLE reader
(`tools/ple_nvme/`, re-pinned to this tree as `0.2.0+systemone4`) + Pennyroyal's
two NIXL bounce-buffer fixes. Only Andrew deploys it, and he qualifies it on
his machine. Its open items are in `madlabs/NIXL.md` on that branch.

It stays a superset of madlabs/main: whoever moves madlabs/main merges it into
`madlabs/main-nixl` the same day, and Andrew may merge madlabs/main into it
whenever he needs. A merge never touches the reader's files, because this line
no longer has them.

## Credits

- **John Pezzulli**: Pennyroyal (jpezzulli/sglang-rtxpro6000), the RTX PRO
  6000 build of Flash-Next that most of the ports above come from. 10
  first-parent commits here carry his authorship.
- **mratsim**: the SM120 online FP8 design, from
  [mratsim/sglang-qwen38fn-sm120-turbo](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo/tree/94a68214b77514bc26ef78cee4c01f128162d09b)
  at `94a68214b7`, patches 0003, 0007 and 0008. Pennyroyal adapted it
  (`FP8.md`:110-117 on `madlabs/pennyroyal-v2.5.0`), and `ce34895c4e` ported
  it here.
- **Andrew Stanton**: System One (the decision routes, calibration and
  benchmarks), and the 2026-09-25 line this trunk grew from. He wrote 51 of
  the 84 commits in `fe89a7c890..922810a5ed`.
- **Garner McCloud**: SSD Stream v0.2.0, which the NVMe reader adapts
  (Apache-2.0; `tools/ple_nvme/ssd_stream/LICENSE` and `NOTICE` on `madlabs/main-nixl`).
- **Upstream authors** whose open or newer work we carry: Justin Huang
  (#39787), Qiaolin Yu (#44150), shi yang (#43032).
