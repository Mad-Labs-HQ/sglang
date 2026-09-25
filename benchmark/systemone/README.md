# System One (`/v1/systemone`) efficacy and speed

Benchmarks for the System One compatible route on off-the-shelf chat models: how
accurate and how calibrated its answers are in each calibration mode, and how
fast it answers many questions about one state. Results below are for
`IFM/K2-Horizon-3.7B` (revision `85f683bc15947495341baa91ae1246dcecc47407`) in
BF16 on one RTX 5070 Ti (16 GB, sm_120) with `--attention-backend flashinfer`.

## Tasks

`tasks.py` samples 600 rows per task with a fixed seed and splits them in half:
the `fit` half is what a fitted calibration or a running batch prior may learn
from, and every metric is on the `test` half (300 rows).

| task | type | question | labels |
| --- | --- | --- | --- |
| `boolq` | noul | the BoolQ question about each passage | balanced |
| `sst2` | noul | Is the sentiment of this movie review positive? | balanced |
| `ag_news_sports` | noul | Is this news article about sports? | skewed, 25% yes |
| `ag_news` | choice | Which topic does this news article cover? (4) | balanced |
| `emotion` | choice | Which emotion does the author express? (6) | skewed |
| `banking77` | choice | Which request does the customer's message make? (77, two-letter labels) | balanced |
| `sst5` | score | How positive is the sentiment? (5 levels) | |
| `yelp` | score | How many stars did the reviewer give? (5 levels) | |

## Scripts

Launch the server with the collection config, which makes every label-free read:

```bash
python -m sglang.launch_server --model-path IFM/K2-Horizon-3.7B \
  --revision 85f683bc15947495341baa91ae1246dcecc47407 --reasoning-parser k2_horizon \
  --attention-backend flashinfer --context-length 32768 --mem-fraction-static 0.85 \
  --disable-prefill-cuda-graph --enable-metrics --port 30000 \
  --decision-calibration-config benchmark/systemone/configs/collect.json
```

| script | what it does |
| --- | --- |
| `collect.py` | Asks every row with `x_calibration=label_free` and `x_return_reads=true`, plus one content-free (`N/A`) read per question, into `runs/` |
| `analyze.py` | Computes every calibration strategy offline from those reads with the server's own functions, applies the label-free selection rule, and writes `efficacy.md`, `efficacy.json`, and the chosen `label_free.json` |
| `ablate_think_tag.py` | Compares K2-Horizon's empty `think`, `think_fast`, and `think_faster` blocks as the answer position |
| `reasoning_arms.py` | Reasons first (medium or high effort), then reads the labels: the System 2 reference |
| `verify_server.py` | Exports the fit half for the fit tool, and checks a server's answers in a mode against its reads |
| `bench_speed.py` | Latency and throughput over state length, questions per request, mode, and load |

```bash
cd benchmark/systemone
python collect.py --url http://127.0.0.1:30000 --rows 600 --concurrency 8 --out runs/k2-horizon-3.7b
python analyze.py --run runs/k2-horizon-3.7b --out results/k2-horizon-3.7b-rtx5070ti
python verify_server.py export --rows 600 --out runs/labeled_fit.jsonl
# Relaunch with --decision-calibration-config examples/runtime/systemone/label_free.json, then:
python -m sglang.srt.entrypoints.systemone.fit_calibration --url http://127.0.0.1:30000 \
  --config ../../examples/runtime/systemone/label_free.json --base label_free \
  --data runs/labeled_fit.jsonl --min-rows 50 --folds 5 --out results/k2-horizon-3.7b-rtx5070ti/fitted.json
python verify_server.py check --url http://127.0.0.1:30000 --mode label_free --rows 600 \
  --concurrency 8 --config ../../examples/runtime/systemone/label_free.json --out results/k2-horizon-3.7b-rtx5070ti/served_label_free.md
```

## Results: K2-Horizon-3.7B on an RTX 5070 Ti

### Summary

- K2-Horizon's chat template always opens a reasoning block, so answers are read
  after the empty `<ifm|think>\n</ifm|think>` block its own replies render.
  Among the empty `think`, `think_fast`, and `think_faster` blocks, `think` has
  the lowest NLL on 6 of 7 tasks (`think_tag_ablation.md`).
- **Raw yes/no reads are nearly useless on this model**: it answers `Yes` or `No`,
  and the lowercase labels carry 0.6% to 6% of the probability. Counting the case
  variants and reading both orders (`label_free`) raises noul accuracy from 0.70
  to 0.86 and lowers ECE from 0.23 to 0.15, with no labeled data.
- Choice and score questions gain nothing from the label-free components kept by
  the selection rule. Fitted profiles are what calibrate them: ECE drops to
  0.04 to 0.10, and to 0.03 to 0.05 on score questions, whose fitted
  temperatures (3.7 to 5.3) show how overconfident raw level reads are.
- Choice rotations raise accuracy by up to 12 points (Banking77 0.60 to 0.72
  with 4 rotations) and lower NLL and Brier on every choice task, but raised the
  emotion task's ECE, so the pre-registered ECE rule left them off.
- Reasoning first ("think then read") raises accuracy by 4 to 24 points on 6 of
  7 tasks (AG News drops 2 to 4 points), at 1.2 to 20 seconds per question
  instead of about 0.01, and makes choice and score answers overconfident: their
  NLL rises from 0.4 to 1.9 to between 1.8 and 5.4.
- Priming the prefix shared by a request's prompts prefills each state once: at
  concurrency 8 over 1k-token states with 16 questions, p50 latency falls from
  3.13 s to 1.46 s and throughput rises from 38 to 83 questions per second.

### Efficacy per task (test half, 300 rows)

`raw` is from the collection run; `label_free` and `fitted` are the server's own
answers in those modes (`served_label_free.md`, `served_fitted.md`), each checked
to equal its reads recombined within 3e-16. `fitted` uses the 15 profiles the fit
tool wrote from the fit halves (`fitted.json`).

| task | type | raw acc / ECE / NLL | label_free acc / ECE / NLL | fitted acc / ECE / NLL |
| --- | --- | --- | --- | --- |
| boolq | noul | 0.663 / 0.277 / 0.676 | 0.810 / 0.138 / 0.472 | 0.827 / 0.080 / 0.451 |
| sst2 | noul | 0.590 / 0.276 / 0.692 | 0.807 / 0.168 / 0.481 | 0.827 / 0.080 / 0.373 |
| ag_news_sports | noul | 0.857 / 0.142 / 0.258 | 0.967 / 0.139 / 0.188 | 0.967 / 0.044 / 0.094 |
| ag_news | choice | 0.890 / 0.044 / 0.336 | 0.890 / 0.040 / 0.341 | 0.887 / 0.042 / 0.333 |
| emotion | choice | 0.460 / 0.200 / 1.709 | 0.460 / 0.197 / 1.705 | 0.467 / 0.079 / 1.482 |
| banking77 | choice | 0.600 / 0.098 / 2.108 | 0.597 / 0.112 / 2.105 | 0.593 / 0.104 / 1.984 |
| sst5 | score | 0.273 / 0.262 / 1.805 | 0.273 / 0.260 / 1.796 | 0.267 / 0.053 / 1.540 |
| yelp | score | 0.277 / 0.178 / 1.790 | 0.263 / 0.186 / 1.790 | 0.260 / 0.027 / 1.573 |

### Label-free components

The rule, fixed before the run: a component is kept when, added to those
already kept, it lowers the mean test ECE over the tasks it applies to and costs
no task more than 1 point of accuracy. Components were tried in this order.

| component | mean ECE change | worst accuracy change | kept |
| --- | --- | --- | --- |
| noul case variants | -0.0507 | +0.0300 | yes |
| noul both orders | -0.0359 | +0.0400 | yes |
| choice name variants | +0.0076 | -0.0033 | no |
| choice rotations 2 / 4 / 8 | +0.0002 / +0.0126 / +0.0079 | +0.0200 / +0.0167 / +0.0167 | no |
| batch prior 0.5 / 0.75 / 1.0 | -0.0094 / -0.0096 / -0.0129 | -0.0167 / -0.0200 / -0.0300 | no |
| content-free prior 0.5 / 1.0 | +0.0307 / +0.0833 | -0.0567 / -0.1233 | no |

Label mass is the full-vocabulary probability of the scored tokens in the read
that shows options in request order:

| task | labels only | with case variants | reads disagree on the top answer |
| --- | --- | --- | --- |
| boolq | 0.006 | 0.925 | 0.340 (2 orders) |
| sst2 | 0.063 | 0.707 | 0.173 (2 orders) |
| ag_news_sports | 0.059 | 0.912 | 0.158 (2 orders) |
| ag_news | 0.202 | | 0.205 (4 rotations) |
| emotion | 0.032 | | 0.542 (6 rotations) |
| banking77 | 0.118 | | 0.712 (8 rotations) |
| sst5 | 0.336 | | |
| yelp | 0.143 | | |

`efficacy.md` has every strategy on every task, including rotations with and
without fitted temperatures.

### Reasoning first

`reasoning_medium.md` and `reasoning_high.md` compare the one-pass read with
reasoning first on the first 50 test rows of each task (Banking77 excluded, as
`/v1/decisions` labels at most 26 options). High effort, 4096-token budget:

| task | one-pass acc | think-then-read acc | one-pass NLL | think-then-read NLL | reasoning tokens | s/question |
| --- | --- | --- | --- | --- | --- | --- |
| boolq | 0.660 | 0.860 | 0.812 | 1.228 | 568 | 8.9 |
| sst2 | 0.740 | 0.920 | 0.510 | 0.550 | 357 | 5.5 |
| ag_news_sports | 0.860 | 0.900 | 0.250 | 0.529 | 442 | 6.8 |
| ag_news | 0.880 | 0.860 | 0.374 | 1.754 | 325 | 5.1 |
| emotion | 0.460 | 0.600 | 1.685 | 5.379 | 669 | 10.3 |
| sst5 | 0.340 | 0.580 | 1.662 | 3.806 | 762 | 11.5 |
| yelp | 0.240 | 0.440 | 1.924 | 4.293 | 1283 | 19.7 |

### Speed

`speed.md` has the full matrix: states of 128 to 12288 tokens, 1 to 64 questions
per request, `raw` and `label_free`, each on three servers: prefix priming on
(`fcfs-prime`, the default), priming off (`fcfs-noprime`,
`SGLANG_DECISION_PREFIX_PRIME_MIN_TOKENS=1000000000`), and priming off with
`--schedule-policy lpm`. One request at a time, p50 seconds:

| state tokens | questions | priming on | priming off | lpm, priming off |
| --- | --- | --- | --- | --- |
| 128 | 16 | 0.267 | 0.262 | 0.262 |
| 1024 | 4 | 0.173 | 0.277 | 0.278 |
| 1024 | 16 | 0.259 | 0.412 | 0.412 |
| 1024 | 64 | 0.595 | 0.791 | 0.784 |
| 4096 | 16 | 0.780 | 0.775 | 0.739 |
| 4096 | 64 | 1.573 | 1.411 | 1.433 |
| 12288 | 16 | 2.484 | 2.505 | 2.488 |

Without priming, the prompts of one request enter the same prefill batch and
each prefills the state, except the part that fills whole prefill chunks
(2048 tokens here), which the first prompt caches for the rest. The 4096- and
12288-token states above are exact multiples of the chunk, so they duplicate
almost nothing, and priming only adds a round trip (up to about 10%). States
of other lengths duplicate their last partial chunk (`speed_unaligned.md`):

| state tokens | questions | priming on | priming off |
| --- | --- | --- | --- |
| 1536 | 4 | 0.234 | 0.513 |
| 1536 | 16 | 0.369 | 0.607 |
| 2560 | 16 | 0.486 | 0.841 |
| 3072 | 16 | 0.582 | 0.766 |
| 3072 | 64 | 1.156 | 1.380 |

Under load (8 concurrent requests, 1024-token states, 16 questions each), p50
is 1.46 s with priming, 3.13 s without, and 2.42 s with `lpm`; throughput is 83,
38, and 50 questions per second.

### Drift

`drift.md` compares the served label-free answers with the same reads from the
collection run, which batched and cached differently:

| task | mean abs diff | p99 abs diff | top answer flips |
| --- | --- | --- | --- |
| boolq | 0.0088 | 0.0309 | 0.007 |
| sst2 | 0.0109 | 0.0403 | 0.023 |
| ag_news_sports | 0.0067 | 0.0288 | 0.007 |
| ag_news | 0.0051 | 0.0751 | 0.007 |
| emotion | 0.0078 | 0.0565 | 0.013 |
| banking77 | 0.0010 | 0.0222 | 0.027 |
| sst5 | 0.0133 | 0.0792 | 0.067 |
| yelp | 0.0100 | 0.0526 | 0.077 |

### Setup notes

- FA3 does not run on sm_120, so the model card's `--attention-backend fa3`
  recipe is replaced by `flashinfer`.
- With the desktop using about 1.2 GB, `--mem-fraction-static 0.85` and
  `--disable-prefill-cuda-graph` leave a KV pool of about 15.5k tokens.
- Label logprobs from `/v1/score` matched the transformers implementation on
  CPU within 0.1 to 0.23 nats (BF16 on both) with the same label ranking.

## Second model: Qwen3-0.6B

Nothing in the route or the calibration is specific to K2-Horizon. The same
collection and analysis on `Qwen/Qwen3-0.6B` (BF16, same GPU, collection config,
`--mem-fraction-static 0.6`), whose chat template has a thinking toggle instead of
an always-open reasoning block, are in `results/qwen3-0.6b-rtx5070ti/`.
Offline strategies on the test half:

| task | type | raw acc / ECE / NLL | label_free | label_free + fitted | choice rotations 4 |
| --- | --- | --- | --- | --- | --- |
| boolq | noul | 0.663 / 0.283 / 1.215 | 0.667 / 0.230 / 0.839 | 0.690 / 0.080 / 0.569 |  |
| sst2 | noul | 0.520 / 0.438 / 1.624 | 0.517 / 0.391 / 1.246 | 0.707 / 0.078 / 0.579 |  |
| ag_news_sports | noul | 0.983 / 0.068 / 0.125 | 0.973 / 0.033 / 0.080 | 0.973 / 0.033 / 0.080 |  |
| ag_news | choice | 0.840 / 0.124 / 0.788 | 0.840 / 0.124 / 0.788 | 0.840 / 0.045 / 0.477 | 0.810 / 0.124 / 0.768 |
| emotion | choice | 0.500 / 0.275 / 2.303 | 0.500 / 0.275 / 2.303 | 0.500 / 0.085 / 1.440 | 0.520 / 0.256 / 1.799 |
| banking77 | choice | 0.073 / 0.485 / 8.200 | 0.073 / 0.485 / 8.200 | 0.073 / 0.034 / 4.258 | 0.237 / 0.091 / 3.828 |
| sst5 | score | 0.220 / 0.766 / 5.092 | 0.220 / 0.766 / 5.092 | 0.220 / 0.063 / 1.604 |  |
| yelp | score | 0.247 / 0.265 / 1.994 | 0.247 / 0.265 / 1.994 | 0.247 / 0.035 / 1.578 |  |

- The selection rule kept the same components as on K2-Horizon, for different
  reasons: rotations lowered choice ECE by up to 0.14 and raised Banking77
  accuracy from 0.07 to 0.24, but cost AG News 3 points, and the batch prior
  lowered ECE by up to 0.16 but cost up to 3.7 points.
- Its letter bias on 77 options is extreme: every Banking77 row changes its top
  answer across rotations, and the raw read is near chance.
- Fitted profiles bring every task's ECE to 0.03 to 0.09, and Platt scaling
  moves the yes or no threshold where the raw read is biased (SST-2 accuracy
  0.52 to 0.71).
