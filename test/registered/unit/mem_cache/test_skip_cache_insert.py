"""skip_cache_insert: a request may opt out of publishing its KV to the prefix
cache while still reading from it.

Covers every hop the flag takes -- GenerateReqInput (and its batch split), the
tokenizer manager, the msgpack IPC hop, the scheduler's Req construction -- and
the two mem_cache/common.py chokepoints that act on it.
"""

import unittest
from array import array
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import TypeAdapter

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req, release_kv_cache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _StopAfterReq(Exception):
    pass


def _tokenize(obj: GenerateReqInput) -> TokenizedGenerateReqInput:
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.preferred_sampling_params = None
    tm.sampling_params_class = SamplingParams
    tm.tokenizer = None
    tm.model_config = SimpleNamespace(vocab_size=128)
    tm.rid_to_state = defaultdict(lambda: SimpleNamespace(time_stats=MagicMock()))
    tokenized = tm._create_tokenized_object(obj, input_text="", input_ids=[1, 2, 3])
    tokenized.time_stats = None  # the mock above does not cross a process
    return tokenized


def _ipc_hop(tokenized: TokenizedGenerateReqInput) -> TokenizedGenerateReqInput:
    tokenized.wrap_pickle_fields()
    decoded = msgpack_decode(msgpack_encode(tokenized))
    decoded.unwrap_pickle_fields()
    return decoded


def _schedule(recv_req: TokenizedGenerateReqInput) -> Req:
    """Run the scheduler's handler just far enough to build its Req."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_session_radix_cache = False
    scheduler.model_config = SimpleNamespace(hf_eos_token_id={1}, vocab_size=128)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.dllm_config = None
    built = []

    def build(*args, **kwargs):
        built.append(Req(*args, **kwargs))
        raise _StopAfterReq

    with (
        patch(
            "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
            return_value=1,
        ),
        patch("sglang.srt.managers.scheduler.Req", side_effect=build),
    ):
        try:
            scheduler.handle_generate_request(recv_req)
        except _StopAfterReq:
            pass
    (req,) = built
    return req


class TestSkipCacheInsertTransport(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_generate_http_body_accepts_the_flag(self):
        """/generate parses its JSON body straight into GenerateReqInput."""
        body = {"input_ids": [1, 2, 3], "skip_cache_insert": True}
        self.assertTrue(
            TypeAdapter(GenerateReqInput).validate_python(body).skip_cache_insert
        )

    def test_defaults_to_publishing(self):
        req = GenerateReqInput(input_ids=[1, 2, 3])
        req.normalize_batch_and_arguments()
        self.assertFalse(req.skip_cache_insert)
        self.assertFalse(_schedule(_ipc_hop(_tokenize(req))).skip_cache_insert)

    def test_flag_reaches_the_scheduler_req(self):
        req = GenerateReqInput(input_ids=[1, 2, 3], skip_cache_insert=True)
        req.normalize_batch_and_arguments()
        tokenized = _tokenize(req)
        self.assertTrue(tokenized.skip_cache_insert)
        decoded = _ipc_hop(tokenized)
        self.assertTrue(decoded.skip_cache_insert)
        self.assertTrue(_schedule(decoded).skip_cache_insert)

    def test_batch_level_flag_applies_to_every_item_and_sample(self):
        req = GenerateReqInput(
            input_ids=[[1, 2], [3, 4]],
            skip_cache_insert=True,
            sampling_params={"n": 2},
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(
            [req[i].skip_cache_insert for i in range(4)], [True, True, True, True]
        )

    def test_is_an_appended_defaulted_wire_field(self):
        """TokenizedGenerateReqInput is positional on the wire and the Rust
        server emits only a prefix of it, so the new field must come last and
        carry a default (a Rust-sent request then decodes as False)."""
        fields = TokenizedGenerateReqInput.__struct_fields__
        self.assertEqual(fields[-1], "skip_cache_insert")
        self.assertIs(TokenizedGenerateReqInput.__struct_defaults__[-1], False)


def _req(*, skip_cache_insert=False, protected=1, committed=3):
    req = Req(
        rid="r",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3]),
        sampling_params=SamplingParams(max_new_tokens=1),
        vocab_size=128,
        skip_cache_insert=skip_cache_insert,
    )
    req.kv = ReqKvInfo(
        req_pool_idx=0,
        kv_committed_len=committed,
        kv_allocated_len=committed,
        cache_protected_len=protected,
    )
    req.last_node = object()  # holds a lock on the matched prefix
    return req


def _tree_cache():
    cache = object.__new__(UnifiedRadixCache)
    cache.cache_controller = None
    cache.session = MagicMock()
    cache.session.try_cache_finished_req.return_value = False
    cache.disable = False
    cache.req_to_token_pool = MagicMock()
    cache.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
    cache.cache_finished_req = MagicMock()
    cache.cache_unfinished_req = MagicMock()
    cache.free_kv_row = MagicMock()
    cache._dec_req_lock = MagicMock()
    cache.component = MagicMock()
    cache._components_tuple = (cache.component,)
    cache.enable_session_radix_cache = False
    return cache


def _release(req, cache, **kwargs):
    with (
        patch(
            "sglang.srt.mem_cache.common.get_spec",
            return_value=SimpleNamespace(speculative_algorithm=None),
        ),
        patch(
            "sglang.srt.mem_cache.common.get_serving",
            return_value=SimpleNamespace(strip_thinking_cache=False),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.get_serving",
            return_value=SimpleNamespace(strip_thinking_cache=False),
        ),
    ):
        release_kv_cache(req, cache, **kwargs)


class TestReleaseWithholdsPublication(CustomTestCase):
    def test_default_request_is_published_at_finish(self):
        req, cache = _req(), _tree_cache()
        _release(req, cache)
        cache.cache_finished_req.assert_called_once_with(req, owned_kv_len=3)
        cache.free_kv_row.assert_not_called()

    def test_opted_out_request_is_freed_not_published(self):
        req, cache = _req(skip_cache_insert=True), _tree_cache()
        _release(req, cache)
        cache.cache_finished_req.assert_not_called()
        # Only what the request owns is freed; the matched (protected) prefix
        # still belongs to the tree.
        cache.free_kv_row.assert_called_once_with(req.kv, [(1, 3)])
        # The prefix lock taken at match time is dropped...
        cache._dec_req_lock.assert_called_once()
        # ...and components (e.g. mamba) free their per-request state.
        cache.component.cleanup_after_caching_req.assert_called_once_with(
            req, is_finished=True
        )
        cache.req_to_token_pool.free.assert_called_once_with(req)
        self.assertTrue(req.kv.is_kv_released)


class TestUnfinishedCaching(CustomTestCase):
    def test_default_request_is_cached_after_prefill(self):
        req, cache = _req(), _tree_cache()
        maybe_cache_unfinished_req(req, cache)
        cache.cache_unfinished_req.assert_called_once_with(req)

    def test_opted_out_request_is_not_cached_after_prefill(self):
        req, cache = _req(skip_cache_insert=True), _tree_cache()
        maybe_cache_unfinished_req(req, cache)
        cache.cache_unfinished_req.assert_not_called()

    def test_opted_out_request_keeps_chunked_prefill_bookkeeping(self):
        """cache_unfinished_req is what advances req.prefix_indices between
        chunks; skipping it would re-prefill the same chunk forever."""
        req, cache = _req(skip_cache_insert=True), _tree_cache()
        maybe_cache_unfinished_req(req, cache, chunked=True)
        cache.cache_unfinished_req.assert_called_once_with(req, chunked=True)

    def test_req_without_init_is_treated_as_default(self):
        req = SimpleNamespace(skip_radix_cache_insert=False)
        cache = _tree_cache()
        maybe_cache_unfinished_req(req, cache)
        cache.cache_unfinished_req.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()
