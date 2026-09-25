"""Labeled System One tasks built from public datasets, one question per row.

Each task asks the same question of every row, as a System One workload would,
except BoolQ, whose question comes with each passage. Rows are sampled with a
fixed seed and split in half: ``fit`` rows for fitted calibration, ``test`` rows
for every reported metric.
"""

import random
from typing import Any, Callable, Dict, List

from datasets import load_dataset

SEED = 0


def _noul(instructions: str) -> Dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def _boolq(row):
    question = row["question"].strip()
    return (
        row["passage"],
        _noul(question[:1].upper() + question[1:] + "?"),
        0 if row["answer"] else 1,
    )


def _sst2(row):
    return (
        row["sentence"].strip(),
        _noul("Is the sentiment of this movie review positive?"),
        0 if row["label"] == 1 else 1,
    )


def _ag_news_sports(row):
    return (
        row["text"],
        _noul("Is this news article about sports?"),
        0 if row["label"] == 1 else 1,
    )


AG_NEWS_TOPICS = {
    "World": "International news, politics, and conflicts",
    "Sports": "Sports events, teams, and athletes",
    "Business": "Companies, markets, and the economy",
    "Science and technology": "Science, computing, and technology products",
}


def _ag_news(row):
    question = {
        "type": "choice",
        "instructions": "Which topic does this news article cover?",
        "criteria": AG_NEWS_TOPICS,
    }
    return row["text"], question, row["label"]


EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def _emotion(row):
    question = {
        "type": "choice",
        "instructions": "Which emotion does the author of this message express?",
        "criteria": {name: None for name in EMOTIONS},
    }
    return row["text"], question, row["label"]


SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def _sst5(row):
    question = {
        "type": "score",
        "instructions": "How positive is the sentiment of this movie review?",
        "criteria": SST5_LEVELS,
    }
    return row["text"], question, row["label"]


YELP_LEVELS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]


def _yelp(row):
    question = {
        "type": "score",
        "instructions": "How many stars did the reviewer give this business?",
        "criteria": YELP_LEVELS,
    }
    return row["text"], question, row["label"]


def _banking77_builder(split_rows) -> Callable:
    intents = sorted({row["label_text"] for row in split_rows})
    names = [intent.replace("_", " ") for intent in intents]
    index = {intent: i for i, intent in enumerate(intents)}
    question = {
        "type": "choice",
        "instructions": "Which request does this bank customer's message make?",
        "criteria": {name: None for name in names},
    }
    return lambda row: (row["text"], question, index[row["label_text"]])


# name -> (dataset, config, split, row builder, question kind, skewed labels)
TASKS = {
    "boolq": ("google/boolq", None, "validation", _boolq, "noul", False),
    "sst2": ("stanfordnlp/sst2", None, "validation", _sst2, "noul", False),
    "ag_news_sports": ("fancyzhx/ag_news", None, "test", _ag_news_sports, "noul", True),
    "ag_news": ("fancyzhx/ag_news", None, "test", _ag_news, "choice", False),
    "emotion": ("dair-ai/emotion", "split", "test", _emotion, "choice", True),
    "banking77": ("mteb/banking77", None, "test", None, "choice", False),
    "sst5": ("SetFit/sst5", None, "test", _sst5, "score", False),
    "yelp": ("Yelp/yelp_review_full", None, "test", _yelp, "score", False),
}


def load_task(name: str, rows: int) -> List[Dict[str, Any]]:
    """Sampled rows of a task: id, split, state, question, and gold option index."""
    dataset, config, split, build, kind, _ = TASKS[name]
    data = load_dataset(dataset, config, split=split)
    if build is None:
        build = _banking77_builder(data)
    picked = random.Random(f"{SEED}:{name}").sample(range(len(data)), rows)
    out = []
    for position, i in enumerate(picked):
        state, question, gold = build(data[i])
        out.append(
            {
                "id": f"{name}:{i}",
                "split": "fit" if position < rows // 2 else "test",
                "kind": kind,
                "state": state,
                "question": question,
                "gold": gold,
            }
        )
    return out
