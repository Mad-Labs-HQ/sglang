"""Latency and throughput of /v1/systemone over state length, question count, and load.

Every request gets its own state, so no request reuses another's prefix: each
point measures a cold state shared only by the questions of one request. Prefill
tokens computed and read from the prefix cache come from the server's /metrics,
so the output shows how much of each state was prefilled more than once.

    python benchmark/systemone/bench_speed.py --url http://127.0.0.1:30000 \\
        --label fcfs-prime --state-tokens 128 1024 4096 --questions 1 4 16 64 \\
        --modes raw label_free --concurrency 1 --requests 8 \\
        --out benchmark/systemone/results/k2-horizon-3.7b-rtx5070ti/speed.jsonl
"""

import argparse
import json
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from datasets import load_dataset
from transformers import AutoTokenizer

TOPICS = [
    "billing",
    "outage",
    "security",
    "shipping",
    "refund",
    "login",
    "pricing",
    "privacy",
]


def corpus_tokens(tokenizer, needed: int):
    """Token ids of real English text, BoolQ passages joined, at least this long."""
    passages = load_dataset("google/boolq", split="validation")["passage"]
    ids = []
    for passage in passages:
        ids.extend(tokenizer.encode(passage + "\n", add_special_tokens=False))
        if len(ids) >= needed:
            return ids
    raise ValueError("corpus too short")


def make_state(tokenizer, corpus, tokens: int, request_index: int) -> str:
    """A state of about this many tokens, starting with text unique to the request."""
    offset = (request_index * 7919) % max(1, len(corpus) - tokens)
    body = tokenizer.decode(corpus[offset : offset + tokens])
    return f"Ticket {request_index}-{random.randrange(10**9)}:\n{body}"


def make_questions(count: int):
    """A mix of noul, 4-option choice, and 5-level score questions."""
    questions = {}
    for i in range(count):
        topic = TOPICS[i % len(TOPICS)]
        kind = ("noul", "choice", "score")[i % 3]
        if kind == "noul":
            question = {
                "type": "noul",
                "instructions": f"Does the text mention {topic} ({i})?",
            }
        elif kind == "choice":
            question = {
                "type": "choice",
                "instructions": f"Which team should read this text first ({i})?",
                "criteria": {
                    "support": None,
                    "engineering": None,
                    "sales": None,
                    "legal": None,
                },
            }
        else:
            question = {
                "type": "score",
                "instructions": f"How relevant is the text to {topic} ({i})?",
                "criteria": ["none", "slight", "some", "high", "complete"],
            }
        questions[f"q{i}"] = question
    return questions


def prefill_counters(url: str):
    """Prefill tokens computed and read from the cache, from /metrics."""
    text = requests.get(f"{url}/metrics", timeout=30).text
    counters = {}
    for mode in ("prefill_compute", "prefill_cache"):
        match = re.search(
            r'^sglang:realtime_tokens_total\{[^}]*mode="' + mode + r'"[^}]*\} (\S+)$',
            text,
            re.M,
        )
        counters[mode] = float(match.group(1)) if match else 0.0
    return counters


def run_point(
    url, tokenizer, corpus, state_tokens, questions, mode, concurrency, requests_n, seed
):
    random.seed(seed)
    bodies = [
        {
            "state": make_state(tokenizer, corpus, state_tokens, seed * 1000 + i),
            "model": "bench",
            "questions": make_questions(questions),
            "x_calibration": mode,
        }
        for i in range(requests_n)
    ]
    session = requests.Session()

    def send(body):
        started = time.perf_counter()
        response = session.post(f"{url}/v1/systemone", json=body, timeout=3600)
        response.raise_for_status()
        return time.perf_counter() - started, response.json()["usage"]["input_tokens"]

    before = prefill_counters(url)
    started = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(send, bodies))
    wall = time.perf_counter() - started
    after = prefill_counters(url)
    latencies = sorted(r[0] for r in results)
    return {
        "state_tokens": state_tokens,
        "questions": questions,
        "mode": mode,
        "concurrency": concurrency,
        "requests": requests_n,
        "p50_s": statistics.median(latencies),
        "p95_s": latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))],
        "mean_s": statistics.fmean(latencies),
        "requests_per_s": requests_n / wall,
        "questions_per_s": requests_n * questions / wall,
        "input_tokens_per_request": statistics.fmean(r[1] for r in results),
        "prefill_computed_tokens": after["prefill_compute"] - before["prefill_compute"],
        "prefill_cached_tokens": after["prefill_cache"] - before["prefill_cache"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--label", required=True, help="server variant, recorded per row"
    )
    parser.add_argument("--state-tokens", type=int, nargs="+", required=True)
    parser.add_argument("--questions", type=int, nargs="+", required=True)
    parser.add_argument(
        "--modes", nargs="+", required=True, choices=["raw", "label_free"]
    )
    parser.add_argument("--concurrency", type=int, nargs="+", required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument(
        "--max-point-tokens",
        type=int,
        required=True,
        help="skip points whose unshared prompt tokens per request exceed this",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    info = requests.get(f"{args.url}/server_info", timeout=30).json()
    tokenizer = AutoTokenizer.from_pretrained(
        info["tokenizer_path"], revision=info["revision"]
    )
    corpus = corpus_tokens(tokenizer, 4 * max(args.state_tokens))
    # One warmup request so the first point does not pay for startup work.
    run_point(args.url, tokenizer, corpus, 64, 2, "raw", 1, 1, seed=99)
    seed = 0
    with open(args.out, "a") as out:
        for state_tokens in args.state_tokens:
            for questions in args.questions:
                if state_tokens * questions > args.max_point_tokens:
                    print(
                        f"skip state={state_tokens} questions={questions}", flush=True
                    )
                    continue
                for mode in args.modes:
                    for concurrency in args.concurrency:
                        seed += 1
                        row = run_point(
                            args.url,
                            tokenizer,
                            corpus,
                            state_tokens,
                            questions,
                            mode,
                            concurrency,
                            args.requests,
                            seed,
                        )
                        row.update(label=args.label, model=info["model_path"])
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        print(
                            f"{args.label} state={state_tokens} q={questions} {mode} c={concurrency}: "
                            f"p50 {row['p50_s']:.3f}s p95 {row['p95_s']:.3f}s "
                            f"{row['questions_per_s']:.1f} q/s computed {row['prefill_computed_tokens']:.0f} "
                            f"cached {row['prefill_cached_tokens']:.0f}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
