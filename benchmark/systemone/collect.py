"""Collect the label logprobs of every read of every task row from /v1/systemone.

Run against a server launched with ``--decision-calibration-config
configs/collect.json``. That config makes the widest set of label-free reads:
up to 8 choice rotations, option-name variants, both noul orders, and case
variants. Every strategy in analyze.py is a function of those reads.
Each question is also asked once with the content-free state ``N/A``, for
contextual calibration.

    python benchmark/systemone/collect.py --url http://127.0.0.1:30000 \\
        --rows 600 --out benchmark/systemone/runs/k2-horizon-3.7b
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from tasks import TASKS, load_task

CONTENT_FREE_STATE = "N/A"


def ask(session, url, state, question):
    started = time.perf_counter()
    response = session.post(
        f"{url}/v1/systemone",
        json={
            "state": state,
            "model": "local",
            "questions": {"q": question},
            "x_calibration": "label_free",
            "x_return_reads": True,
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    return body["answers"]["q"], body["usage"], time.perf_counter() - started


def question_key(question):
    return json.dumps(question, sort_keys=True, ensure_ascii=False)


def collect_task(url, name, rows, out_dir, concurrency):
    data = load_task(name, rows)
    session = requests.Session()

    def run(row):
        answer, usage, latency = ask(session, url, row["state"], row["question"])
        return {
            **{k: row[k] for k in ("id", "split", "kind", "gold", "question")},
            "answer": answer,
            "input_tokens": usage["input_tokens"],
            "latency": latency,
        }

    started = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(run, data))
    elapsed = time.perf_counter() - started
    with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    questions = {question_key(row["question"]): row["question"] for row in data}
    with ThreadPoolExecutor(concurrency) as pool:
        content_free = list(
            pool.map(
                lambda q: ask(session, url, CONTENT_FREE_STATE, q)[0],
                questions.values(),
            )
        )
    with open(os.path.join(out_dir, f"{name}.content_free.jsonl"), "w") as f:
        for key, answer in zip(questions, content_free):
            f.write(
                json.dumps({"key": key, "answer": answer}, ensure_ascii=False) + "\n"
            )
    tokens = sum(r["input_tokens"] for r in results)
    print(
        f"{name}: {len(results)} rows in {elapsed:.1f}s, {tokens} input tokens, "
        f"{len(questions)} content-free questions",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    info = requests.get(f"{args.url}/server_info", timeout=30).json()
    with open(os.path.join(args.out, "server.json"), "w") as f:
        json.dump(
            {k: info.get(k) for k in ("model_path", "revision", "attention_backend")},
            f,
            indent=2,
        )
    for name in args.tasks:
        collect_task(args.url, name, args.rows, args.out, args.concurrency)


if __name__ == "__main__":
    main()
