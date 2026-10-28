from types import SimpleNamespace

import pytest
from sglang_ssd_stream.qwen4 import Qwen4PLELayer

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    enable_breakable_cuda_graph,
)


def test_prefetch_refuses_a_breakable_cuda_graph():
    # SGLang's breakable backend sets this flag for warmup, capture and replay.
    layer = SimpleNamespace(_prefetch_state=None)
    with (
        pytest.raises(RuntimeError, match="breakable CUDA graphs are not supported"),
        enable_breakable_cuda_graph(),
    ):
        Qwen4PLELayer.start_prefetch(layer, batch=None, forward_batch=None)


def test_prefetch_check_is_keyed_on_the_breakable_flag():
    # Outside a breakable graph the prefetch goes on to its own checks.
    layer = SimpleNamespace(_prefetch_state=object())
    with pytest.raises(RuntimeError, match="was not consumed before reuse"):
        Qwen4PLELayer.start_prefetch(layer, batch=None, forward_batch=None)
