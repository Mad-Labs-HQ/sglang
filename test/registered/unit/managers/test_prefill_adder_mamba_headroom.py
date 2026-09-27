"""CPU-only tests for the separate Mamba pool's admission gate.

With a separate Mamba pool (`HybridReqToTokenPool`'s own allocator) nothing in
`PrefillAdder` used to budget Mamba slots, so a request admitted while most of
the pool was pinned took the scheduler down in `prepare_for_extend`:

  AssertionError: Not enough space for mamba ping pong idx
    memory_pool.py _alloc_ping_pong_buffer <- HybridReqToTokenPool.alloc
    <- alloc_req_slots <- alloc_for_extend <- prepare_for_extend
    <- _get_new_batch_prefill_raw

Seen in production (32 slots, 6 running, extra_buffer + overlap): the last
chunk of a long chunked prefill builds a write-through backup chain that pins
the un-backed Mamba state of every chunk until the D2H ack, taking usage from
0.62 to 0.94 in one step; the next new request asserted. `try_reserve_mamba`
now defers such a request to the next step instead.

These tests drive a real `HybridReqToTokenPool` on CPU. On the old code the
gate does not exist, so every test that calls it errors.
"""

import random
import unittest
from array import array
from unittest.mock import MagicMock

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PING_PONG_ASSERT = "Not enough space for mamba ping pong idx"


def _make_pool(
    mamba_size: int,
    *,
    max_reqs: int = 8,
    extra_buffer: bool = True,
    lazy: bool = False,
) -> HybridReqToTokenPool:
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=64,
        n_groups=1,
        num_heads=2,
        head_dim=32,
        state_size=8,
        conv_kernel=4,
    )
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        params = Mamba2CacheParams(shape=shape, layers=[0])
    return HybridReqToTokenPool(
        size=max_reqs,
        mamba_size=mamba_size,
        mamba_spec_state_size=max_reqs,
        max_context_len=64,
        device="cpu",
        enable_memory_saver=False,
        cache_params=params,
        mamba_layer_ids=[0],
        enable_mamba_extra_buffer=extra_buffer,
        enable_mamba_extra_buffer_lazy=lazy,
        speculative_num_draft_tokens=4,
        enable_overlap_schedule=True,
    )


def _make_req(rid) -> Req:
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3]),
        sampling_params=SamplingParams(max_new_tokens=1),
    )


class TestMambaAdmissionHeadroom(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _make_adder(self, pool, *, evictable=lambda: 0, int8_ckpt=False):
        tree_cache = MagicMock()
        tree_cache.full_evictable_size.return_value = 0
        tree_cache.swa_evictable_size.return_value = 0
        tree_cache.evictable_size.return_value = 0
        tree_cache.disable = False
        tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        tree_cache.buffer_pipeline = None
        tree_cache.supports_mamba.return_value = True
        tree_cache.mamba_evictable_size.side_effect = evictable
        tree_cache.req_to_token_pool = pool
        if int8_ckpt:
            pool.mamba_ckpt_pool = object()

        allocator = MagicMock()
        allocator.full_available_size.return_value = 1_000_000
        allocator.available_size.return_value = 1_000_000
        allocator.size_swa = 1_000_000
        allocator.swa_req_ring = False
        allocator.create_prefill_budget.side_effect = lambda tc, **kw: PrefillBudget(
            allocator, tc, **kw
        )
        running_batch = MagicMock()
        running_batch.reqs = []
        return PrefillAdder(
            page_size=1,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=1_000_000,
            rem_chunk_tokens=None,
        )

    def test_crash_state_asserts_in_alloc(self):
        """The production failure, replayed: 2 free slots, a fresh request needs
        1 state + 2 ping-pong. This is what the gate has to keep from happening."""
        pool = _make_pool(8)
        pool.mamba_allocator.alloc(6)  # pinned by a write-through backup chain
        with self.assertRaisesRegex(AssertionError, PING_PONG_ASSERT):
            pool.alloc([_make_req("new")])

    def test_crash_state_defers_before_any_allocation(self):
        pool = _make_pool(8)
        pool.mamba_allocator.alloc(6)
        adder = self._make_adder(pool)
        self.assertFalse(adder.try_reserve_mamba(_make_req("new")))
        self.assertEqual(pool.mamba_allocator.available_size(), 2)
        self.assertEqual(adder.mamba_state_reserved, 0)

    def test_admission_resumes_once_the_pins_release(self):
        pool = _make_pool(8)
        pinned = pool.mamba_allocator.alloc(6)
        req = _make_req("new")
        self.assertFalse(self._make_adder(pool).try_reserve_mamba(req))
        pool.mamba_allocator.free(pinned)  # the D2H acks drained
        self.assertTrue(self._make_adder(pool).try_reserve_mamba(req))
        pool.alloc([req])  # and the allocation it reserved for succeeds

    def test_reservations_accumulate_within_a_round(self):
        pool = _make_pool(8)
        adder = self._make_adder(pool)
        self.assertEqual(adder.mamba_state_headroom, 8)
        self.assertTrue(adder.try_reserve_mamba(_make_req("a")))
        self.assertTrue(adder.try_reserve_mamba(_make_req("b")))
        self.assertEqual(adder.mamba_state_reserved, 8)
        self.assertFalse(adder.try_reserve_mamba(_make_req("c")))

    def test_evictable_states_count_as_headroom(self):
        pool = _make_pool(8)
        pool.mamba_allocator.alloc(6)
        adder = self._make_adder(pool, evictable=lambda: 2)
        self.assertEqual(adder.mamba_state_headroom, 4)
        self.assertTrue(adder.try_reserve_mamba(_make_req("new")))

    def test_int8_checkpoint_states_free_nothing_in_the_active_pool(self):
        pool = _make_pool(8)
        pool.mamba_allocator.alloc(6)
        adder = self._make_adder(pool, evictable=lambda: 2, int8_ckpt=True)
        self.assertEqual(adder.mamba_state_headroom, 2)
        self.assertFalse(adder.try_reserve_mamba(_make_req("new")))

    def test_cost_per_request(self):
        pool = _make_pool(16)
        adder = self._make_adder(pool)
        fresh = _make_req("fresh")
        self.assertEqual(adder._mamba_state_cost(fresh), 4)  # state + 2 + lock

        holds_state = _make_req("holds")
        holds_state.kv.mamba_pool_idx = pool.mamba_allocator.alloc(1)[0]
        self.assertEqual(adder._mamba_state_cost(holds_state), 3)

        continuing = _make_req("continuing")
        pool.alloc([continuing])
        self.assertEqual(adder._mamba_state_cost(continuing), 0)
        self.assertTrue(adder.try_reserve_mamba(continuing))
        self.assertEqual(adder.mamba_state_reserved, 0)

    def test_cost_lazy_and_without_extra_buffer(self):
        lazy = self._make_adder(_make_pool(8, lazy=True))
        self.assertEqual(lazy._mamba_state_cost(_make_req("r")), 3)
        plain = self._make_adder(_make_pool(8, extra_buffer=False))
        self.assertEqual(plain._mamba_state_cost(_make_req("r")), 2)

    def test_inactive_without_a_separate_mamba_pool(self):
        adder = self._make_adder(MagicMock())  # not a HybridReqToTokenPool
        self.assertIsNone(adder.mamba_state_headroom)
        self.assertTrue(adder.try_reserve_mamba(_make_req("r")))

    def test_admitted_rounds_never_assert(self):
        """Whatever the gate admits, the real allocations succeed.

        Replays one admission round against random pool states: pinned slots
        (running requests, chain-locked states), evictable tree states, and a
        mix of misses, device prefix hits (COW: one new slot, and admission
        locks the source state out of the evictable set) and host hits
        (load-back: the request's slot plus the restored node's). Evictable
        states are evicted on demand, as `alloc_req_slots` and the COW path do.
        """
        rng = random.Random(0)
        admitted_total = deferred_total = 0
        for trial in range(300):
            size = rng.randint(4, 16)
            pool = _make_pool(size)
            pinned_n = rng.randint(0, size)
            evictable_n = rng.randint(0, size - pinned_n)
            if pinned_n:
                pool.mamba_allocator.alloc(pinned_n)
            evictable = (
                list(pool.mamba_allocator.alloc(evictable_n).split(1))
                if evictable_n
                else []
            )
            adder = self._make_adder(pool, evictable=lambda: len(evictable))

            def alloc_one():
                slot = pool.mamba_allocator.alloc(1)
                if slot is None and evictable:
                    pool.mamba_allocator.free(evictable.pop())
                    slot = pool.mamba_allocator.alloc(1)
                self.assertIsNotNone(slot, f"trial {trial}: admitted but no slot")
                return slot

            admitted = []
            for i in range(rng.randint(1, 4)):
                req = _make_req(f"{trial}-{i}")
                if not adder.try_reserve_mamba(req):
                    deferred_total += 1
                    break
                kind = rng.choice(("miss", "device_hit", "host_hit"))
                if kind == "device_hit" and evictable:
                    evictable.pop()  # the COW source, now locked by admission
                    req.kv.mamba_pool_idx = alloc_one()[0]
                elif kind == "host_hit":
                    alloc_one()  # the restored tree node, locked until the ack
                    req.kv.mamba_pool_idx = alloc_one()[0]
                admitted.append(req)

            if admitted:
                need = sum(
                    (0 if r.kv.holds_mamba else 1) + pool.mamba_ping_pong_track_buffer_size
                    for r in admitted
                )
                while pool.mamba_allocator.available_size() < need and evictable:
                    pool.mamba_allocator.free(evictable.pop())
                pool.alloc(admitted)  # must not assert
                admitted_total += len(admitted)

        # The replay exercised both outcomes.
        self.assertGreater(admitted_total, 0)
        self.assertGreater(deferred_total, 0)


if __name__ == "__main__":
    unittest.main()
