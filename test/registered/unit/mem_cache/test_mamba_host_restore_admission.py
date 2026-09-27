"""CPU-only tests for charging a HiCache host hit's restored Mamba state.

`Scheduler._ensure_mamba_admission_capacity` (sgl-project/sglang#39787) gates
each candidate on `mamba_admission_slots`: its own state plus its initial
ping-pong buffers. A host hit takes one more slot. `prepare_load_back` binds
the request's state, and the controller then restores the cached node's state
into another device slot, locked until the load completes. Admitted into an
exactly-fitting pool, the load-back leaves too little for the ping-pong
buffers and `alloc_req_slots` fails loud:

  RuntimeError: alloc_req_slots runs out of memory

That is the same scheduler outage as the original "Not enough space for mamba
ping pong idx" assert, reached through another exception. It is what
production hits: a write-through backup chain pins most of a 32-slot pool
while HiCache serves host hits. `PrefillAdder._mamba_covers_host_restore`
charges the restore after the match, before anything is loaded.
"""

import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocation import alloc_req_slots
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

OUT_OF_MEMORY = "alloc_req_slots runs out of memory"


def _pool(mamba_size: int, req_rows: int = 8) -> HybridReqToTokenPool:
    pool = object.__new__(HybridReqToTokenPool)
    ReqToTokenPool.__init__(
        pool, size=req_rows, max_context_len=8, device="cpu", enable_memory_saver=False
    )
    pool.mamba_ping_pong_track_buffer_size = 2
    pool.enable_mamba_extra_buffer = True
    pool.enable_mamba_extra_buffer_lazy = False
    pool.mamba_allocator = MambaSlotAllocator(size=mamba_size, device="cpu")
    pool.mamba_ckpt_pool = None
    pool.mamba_pool = SimpleNamespace(
        replayssm_spec_write_pos=None, replayssm_write_pos=None
    )
    pool.req_index_to_mamba_index_mapping = torch.zeros(req_rows + 1, dtype=torch.int32)
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
        (req_rows + 1, 2), dtype=torch.int64
    )
    return pool


class _Tree:
    """Evictable Mamba states are allocated slots that eviction frees."""

    def __init__(self, pool: HybridReqToTokenPool, evictable: int = 0):
        self.req_to_token_pool = pool
        self.evictable = (
            list(pool.mamba_allocator.alloc(evictable).split(1)) if evictable else []
        )

    def supports_mamba(self) -> bool:
        return True

    def mamba_evictable_size(self) -> int:
        return len(self.evictable)

    def evict_for_alloc(self, params) -> None:
        for _ in range(min(params.mamba_num, len(self.evictable))):
            self.req_to_token_pool.mamba_allocator.free(self.evictable.pop())

    def get_session_kv(self, req):
        return None


def _req(rid, host_hit: int = 0):
    return SimpleNamespace(
        rid=rid, kv=ReqKvInfo(), mamba_host_hit_length=host_hit, finished=lambda: False
    )


def _scheduler(tree: _Tree):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.req_to_token_pool = tree.req_to_token_pool
    scheduler.token_to_kv_pool_allocator = SimpleNamespace()
    scheduler.tree_cache = tree
    return scheduler


def _covers_restore(tree: _Tree, req, admitted=()) -> bool:
    adder = SimpleNamespace(
        tree_cache=tree,
        token_to_kv_pool_allocator=SimpleNamespace(),
        can_run_list=list(admitted),
    )
    return PrefillAdder._mamba_covers_host_restore(adder, req)


def _alloc_one(tree: _Tree):
    """One slot, evicting one state first if needed (the COW / load-back path)."""
    allocator = tree.req_to_token_pool.mamba_allocator
    slot = allocator.alloc(1)
    if slot is None:
        tree.evict_for_alloc(SimpleNamespace(mamba_num=1))
        slot = allocator.alloc(1)
    return slot


def _load_back(tree: _Tree, req) -> bool:
    """`load_back`: bind the request's state, then restore the node's state."""
    dst = None
    if not req.kv.holds_mamba:
        dst = _alloc_one(tree)
        assert dst is not None, "Cannot alloc mamba for load_back"
        req.kv.mamba_pool_idx = dst[0]
    for _ in range(req.mamba_host_hit_length):
        if _alloc_one(tree) is None:  # controller: atomic rollback, load fails
            if dst is not None:
                tree.req_to_token_pool.mamba_allocator.free(dst)
                req.kv.mamba_pool_idx = None
            return False
    return True


class TestMambaHostRestoreAdmission(CustomTestCase):
    def test_exact_fit_host_hit_passes_the_pre_match_gate_then_fails_loud(self):
        """The gap in the pre-match accounting alone, replayed end to end."""
        pool = _pool(mamba_size=3)
        tree = _Tree(pool)
        req = _req("host", host_hit=1)
        self.assertTrue(
            Scheduler._ensure_mamba_admission_capacity(_scheduler(tree), req, [])
        )
        self.assertTrue(_load_back(tree, req))
        with self.assertRaisesRegex(RuntimeError, OUT_OF_MEMORY):
            alloc_req_slots(pool, [req], tree)

    def test_restore_charge_defers_before_anything_is_loaded(self):
        pool = _pool(mamba_size=3)
        tree = _Tree(pool)
        req = _req("host", host_hit=1)
        self.assertFalse(_covers_restore(tree, req))
        self.assertEqual(pool.mamba_allocator.available_size(), 3)
        self.assertIsNone(req.kv.mamba_pool_idx)

    def test_room_for_the_restore_admits_and_allocates(self):
        pool = _pool(mamba_size=4)
        tree = _Tree(pool)
        req = _req("host", host_hit=1)
        self.assertTrue(_covers_restore(tree, req))
        self.assertTrue(_load_back(tree, req))
        alloc_req_slots(pool, [req], tree)
        self.assertEqual(pool.mamba_allocator.available_size(), 0)

    def test_evicts_to_cover_the_restore(self):
        pool = _pool(mamba_size=4)
        tree = _Tree(pool, evictable=1)
        req = _req("host", host_hit=1)
        self.assertTrue(_covers_restore(tree, req))
        self.assertEqual(tree.mamba_evictable_size(), 0)
        self.assertEqual(pool.mamba_allocator.available_size(), 4)

    def test_counts_requests_already_admitted_this_round(self):
        """An admitted, COW'd request still needs its 2 ping-pong slots; the
        host hit needs 1 + 2 + 1 on top."""
        for mamba_size, covered in ((6, False), (7, True)):
            pool = _pool(mamba_size=mamba_size)
            admitted = _req("cowed")
            admitted.kv.mamba_pool_idx = pool.mamba_allocator.alloc(1)[0]
            req = _req("host", host_hit=1)
            self.assertEqual(
                _covers_restore(_Tree(pool), req, [admitted]), covered, mamba_size
            )

    def test_inactive_without_a_host_hit_or_a_hybrid_pool(self):
        pool = _pool(mamba_size=1)
        self.assertTrue(_covers_restore(_Tree(pool), _req("miss")))
        not_hybrid = SimpleNamespace(req_to_token_pool=object())
        self.assertTrue(_covers_restore(not_hybrid, _req("host", host_hit=1)))

    def _replay_rounds(self, trials: int, seed: int):
        """Admission rounds against random pool states: pinned slots (running
        requests, chain-locked backups), evictable tree states, and a mix of
        misses, device hits (COW; admission locks the source out of the
        evictable set) and host hits (load-back + restored node). Mirrors the
        scheduler loop: pre-match gate (skip), match, restore check (stop),
        then `alloc_req_slots` for the round."""
        rng = random.Random(seed)
        admitted_total = deferred_total = 0
        for trial in range(trials):
            size = rng.randint(4, 16)
            pool = _pool(mamba_size=size)
            pinned = rng.randint(0, size)
            if pinned:
                pool.mamba_allocator.alloc(pinned)
            tree = _Tree(pool, evictable=rng.randint(0, size - pinned))
            scheduler = _scheduler(tree)
            admitted = []
            for i in range(rng.randint(1, 4)):
                req = _req(f"{trial}-{i}")
                if not Scheduler._ensure_mamba_admission_capacity(
                    scheduler, req, admitted
                ):
                    deferred_total += 1
                    continue
                kind = rng.choice(("miss", "device_hit", "host_hit"))
                if kind == "device_hit" and tree.evictable:
                    tree.evictable.pop()  # the COW source, locked by admission
                    slot = _alloc_one(tree)
                    self.assertIsNotNone(slot, f"trial {trial}: COW has no slot")
                    req.kv.mamba_pool_idx = slot[0]
                elif kind == "host_hit":
                    req.mamba_host_hit_length = 1
                    if not _covers_restore(tree, req, admitted):
                        deferred_total += 1
                        break
                    if not _load_back(tree, req):
                        break
                admitted.append(req)
            if admitted:
                alloc_req_slots(pool, admitted, tree)  # must not fail loud
                admitted_total += len(admitted)
        return admitted_total, deferred_total

    def test_admitted_rounds_always_allocate(self):
        admitted, deferred = self._replay_rounds(trials=400, seed=0)
        self.assertGreater(admitted, 0)
        self.assertGreater(deferred, 0)

    def test_the_replay_needs_the_restore_charge(self):
        """Without the post-match charge, the same replay fails loud."""
        original = PrefillAdder._mamba_covers_host_restore
        PrefillAdder._mamba_covers_host_restore = lambda self, req: True
        try:
            with self.assertRaisesRegex(RuntimeError, OUT_OF_MEMORY):
                self._replay_rounds(trials=400, seed=0)
        finally:
            PrefillAdder._mamba_covers_host_restore = original


if __name__ == "__main__":
    unittest.main()
