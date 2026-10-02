"""Measurement trace for the hybrid (Mamba) prefix cache. SPIKE CODE.

Answers one question about the deployed engine: when a returning conversation
misses, did its token records (KV) go, or only its Mamba state, and which pool
dropped that state, for what reason? Every Mamba state the cache creates,
copies to host or drops is logged with the node, its depth and the cause, and
every request's match is logged with the longest KV run it could have reused.

Off unless SGLANG_MADLABS_CACHE_TRACE=1. One "MLTRACE {json}" line per event.
"""

import json
import logging
import os
import time
from contextlib import contextmanager

ENABLED = os.environ.get("SGLANG_MADLABS_CACHE_TRACE") == "1"
logger = logging.getLogger(__name__)
_causes = ["other"]


def emit(ev, **fields):
    if ENABLED:
        record = {"t": round(time.time(), 3), "ev": ev, "cause": _causes[-1], **fields}
        logger.info("MLTRACE %s", json.dumps(record, separators=(",", ":")))


@contextmanager
def cause(name):
    """Attribute the evictions inside this block to `name`."""
    _causes.append(name)
    try:
        yield
    finally:
        _causes.pop()


def push_cause(name):
    _causes.append(name)


def pop_cause():
    if len(_causes) > 1:
        _causes.pop()


def depth(node):
    """Token depth of a radix node: the length of the prefix it ends."""
    total = 0
    while node is not None and node.parent is not None:
        total += len(node.key)
        node = node.parent
    return total
