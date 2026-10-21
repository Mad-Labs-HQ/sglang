"""Qwen4 PLE prefill: the packed short conv against the padded one it replaces.

The gate is bitwise: the same batch goes through _prepare_ple_batch and
_short_conv once with SGLANG_ENABLE_QWEN4_PLE_FUSION off (padded) and once
with it on (packed), and outputs and state slots must match bit for bit. The
memory tests bound what the packed path allocates for a skewed batch.
"""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.mamba import qwen4_short_conv
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen4_exp
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")

# Qwen3.8-Flash-Next: hidden 2560 x 4 hyper-connections, kernel 4, dilation 3.
HIDDEN = 2560
HC_COUNT = 4
CHANNELS = HIDDEN * HC_COUNT
KERNEL = 4
DILATION = 3
STATE_LEN = (KERNEL - 1) * DILATION
MiB = 1 << 20
SEEDS = range(5)

# The probe batch that OOMed, its 4-row variant, the chunk-4096 worst case,
# one-token rows, rows shorter than the state, and empty rows (leading,
# middle and trailing).
CASES = (
    [3860] + [40] * 5,
    [3860] + [40] * 3,
    [4091] + [1] * 5,
    [1] * 6,
    [2, 3, 8, 9, 10],
    [0, 5, 0, 12, 0, 0],
    [17, 0],
)

# Track offsets per row, cycled: none, mid-row, the row end, masked off, and
# values the clamp moves to 0 and to the row end.
TRACK_KINDS = ("zero", "mid", "len", "masked", "below", "above")

_BIT_VIEWS = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _generator(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(shape, dtype, generator):
    return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)


def _sprinkle_negative_zeros(tensor, generator):
    hit = torch.rand(tensor.shape, device="cuda", generator=generator) < 0.25
    return torch.where(hit, torch.full_like(tensor, -0.0), tensor)


def _conv_layer(channels, generator):
    layer = qwen4_exp.Qwen4ExpPLELayer.__new__(qwen4_exp.Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.layer_id = 0
    layer.conv_channels = channels
    layer.short_conv_dilation = DILATION
    layer.short_conv_state_len = STATE_LEN
    layer.conv1d = nn.Conv1d(
        channels,
        channels,
        KERNEL,
        groups=channels,
        padding=STATE_LEN,
        dilation=DILATION,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    layer.conv1d.weight.requires_grad_(False)
    layer.conv1d.weight.copy_(
        _randn(layer.conv1d.weight.shape, torch.bfloat16, generator)
    )
    return layer


def _track(lengths, track_slots):
    mask, aligned = [], []
    for row, length in enumerate(lengths):
        kind = TRACK_KINDS[row % len(TRACK_KINDS)]
        # For an empty row the padded path copies the dummy slot into the
        # track slot while the packed path writes nothing (covered by
        # test_qwen4_short_conv), so the gate keeps empty rows masked.
        mask.append(kind != "masked" and length > 0)
        aligned.append(
            {
                "zero": 0,
                "mid": length // 2,
                "len": length,
                "masked": length // 2,
                "below": -7,
                "above": length + 64,
            }[kind]
        )
    return track_slots, mask, aligned


def _forward_batch(
    lengths, slots, *, track=None, processed_tokens=None, track_dtype=torch.int64
):
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        _original_forward_mode=None,
        tbo_parent_token_range=None,
        spec_algorithm=None,
        spec_info=None,
        global_num_token_non_padded_cpu=processed_tokens,
        extend_seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        extend_seq_lens_cpu=list(lengths),
        req_pool_indices=torch.tensor(slots, dtype=torch.int64, device="cuda"),
        out_cache_loc=None,
        mamba_track_indices=None,
        mamba_track_mask=None,
        mamba_track_aligned_lens=lambda: None,
    )
    if track is not None:
        track_slots, mask, aligned = track
        aligned = torch.tensor(aligned, dtype=track_dtype, device="cuda")
        batch.mamba_track_indices = torch.tensor(
            track_slots, dtype=track_dtype, device="cuda"
        )
        batch.mamba_track_mask = torch.tensor(mask, dtype=torch.bool, device="cuda")
        batch.mamba_track_aligned_lens = lambda: aligned
    return batch


def _pool(conv_state):
    return SimpleNamespace(
        ple_window_cache=None,
        get_mamba_indices=lambda indices: indices,
        short_conv_layer_cache=lambda layer_id: conv_state,
    )


class _Path:
    """Patches one PLE path in and counts packed-kernel calls."""

    def __init__(self, conv_state, *, packed):
        self._patches = (
            patch.object(
                qwen4_exp, "get_req_to_token_pool", return_value=_pool(conv_state)
            ),
            patch.object(
                qwen4_exp.envs.SGLANG_ENABLE_QWEN4_PLE_FUSION,
                "get",
                return_value=packed,
            ),
            patch.object(
                qwen4_short_conv,
                "fused_qwen4_varlen_conv",
                wraps=qwen4_short_conv.fused_qwen4_varlen_conv,
            ),
        )
        self.packed_calls = None

    def __enter__(self):
        self._grad = torch.no_grad()
        self._grad.__enter__()
        mocks = [p.__enter__() for p in self._patches]
        self.packed_calls = mocks[2]
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)
        self._grad.__exit__(*exc)
        return False


def _prepare(forward_batch, physical_tokens):
    return qwen4_exp._prepare_ple_batch(
        torch.zeros(physical_tokens, dtype=torch.int64, device="cuda"),
        forward_batch,
        ngram_size=None,
        ngram_eos_token_id=None,
    )


def _short_conv(layer, x, forward_batch, conv_state, *, packed):
    with _Path(conv_state, packed=packed) as path:
        batch = _prepare(forward_batch, x.shape[0])
        output = layer._short_conv(x, forward_batch, batch)
        calls = path.packed_calls.call_count
    return output, calls


class _BitwiseMixin:
    def assert_bits_equal(self, actual, expected, what):
        self.assertEqual(actual.dtype, expected.dtype, what)
        self.assertEqual(actual.shape, expected.shape, what)
        view = _BIT_VIEWS[actual.dtype]
        differ = actual.view(view) != expected.view(view)
        self.assertFalse(
            bool(differ.any()),
            f"{what}: {int(differ.sum())} of {differ.numel()} values differ",
        )


@requires_cuda
class TestQwen4PackedPrefillBitwise(_BitwiseMixin, CustomTestCase):
    def _compare(
        self,
        lengths,
        *,
        seed,
        channels=CHANNELS,
        track=False,
        state_dtype=torch.bfloat16,
        negative_zeros=False,
        track_dtype=torch.int64,
    ):
        generator = _generator(seed)
        rows = len(lengths)
        main_slots = list(range(1, rows + 1))
        track_slots = list(range(rows + 1, 2 * rows + 1))
        layer = _conv_layer(channels, generator)
        x = _randn((sum(lengths), channels), torch.bfloat16, generator)
        state = _randn((2 * rows + 1, channels, STATE_LEN), state_dtype, generator)
        if negative_zeros:
            x = _sprinkle_negative_zeros(x, generator)
            state = _sprinkle_negative_zeros(state, generator)
            layer.conv1d.weight.copy_(
                _sprinkle_negative_zeros(layer.conv1d.weight, generator)
            )
        track_spec = _track(lengths, track_slots) if track else None

        padded_state = state.clone()
        expected, padded_calls = _short_conv(
            layer,
            x,
            _forward_batch(
                lengths, main_slots, track=track_spec, track_dtype=track_dtype
            ),
            padded_state,
            packed=False,
        )
        packed_state = state.clone()
        actual, packed_calls = _short_conv(
            layer,
            x,
            _forward_batch(
                lengths, main_slots, track=track_spec, track_dtype=track_dtype
            ),
            packed_state,
            packed=True,
        )

        self.assertEqual(padded_calls, 0)
        self.assertEqual(packed_calls, 1, "the packed kernel did not run")
        self.assert_bits_equal(actual, expected, "output")
        # Slot 0 is the reserved dummy: the padded path scribbles on it, the
        # packed path must leave it alone.
        self.assert_bits_equal(packed_state[1:], padded_state[1:], "state slots")
        self.assert_bits_equal(packed_state[0], state[0], "dummy slot 0")
        return padded_state

    def test_packed_matches_padded_bitwise(self):
        for lengths in CASES:
            for track in (False, True):
                for seed in SEEDS:
                    with self.subTest(lengths=lengths, track=track, seed=seed):
                        self._compare(lengths, seed=seed, track=track)

    def test_int32_track_metadata_matches_bitwise(self):
        # Replay reads track slots and seqlens from int32 registry buffers; the packed
        # kernel takes int64 only, so the call site must normalise rather than raise.
        for lengths in CASES:
            with self.subTest(lengths=lengths):
                self._compare(
                    lengths, seed=SEEDS[0], track=True, track_dtype=torch.int32
                )

    def test_fp32_state_matches_bitwise(self):
        for lengths in ([2, 3, 8, 9, 10], [0, 5, 0, 12, 0, 0], [3860] + [40] * 5):
            for track in (False, True):
                for seed in SEEDS:
                    with self.subTest(lengths=lengths, track=track, seed=seed):
                        self._compare(
                            lengths,
                            seed=seed,
                            track=track,
                            state_dtype=torch.float32,
                        )

    def test_negative_zeros_keep_their_sign(self):
        # Rows shorter than the state carry old columns into the new state,
        # which is where a -0.0 can lose its sign.
        for state_dtype in (torch.bfloat16, torch.float32):
            sign_bit = torch.tensor([-0.0], dtype=state_dtype, device="cuda").view(
                _BIT_VIEWS[state_dtype]
            )
            for track in (False, True):
                for seed in SEEDS:
                    with self.subTest(state_dtype=state_dtype, track=track, seed=seed):
                        new_state = self._compare(
                            [2, 3, 8, 9, 10],
                            seed=seed,
                            track=track,
                            state_dtype=state_dtype,
                            negative_zeros=True,
                        )
                        stored = new_state[1:].view(_BIT_VIEWS[state_dtype])
                        self.assertTrue(
                            bool((stored == sign_bit).any()),
                            "no -0.0 reached the state; the case does not bite",
                        )

    def test_partial_channel_block(self):
        # 257 channels leave one channel in the last 128- and 64-wide blocks.
        for lengths in ([2, 3, 8, 9, 10], [1] * 6, [0, 5, 0, 12, 0, 0], [130, 7]):
            for track in (False, True):
                for seed in SEEDS:
                    with self.subTest(lengths=lengths, track=track, seed=seed):
                        self._compare(lengths, seed=seed, channels=257, track=track)

    def test_chunked_prefill_matches_one_shot(self):
        # Chunk 2048 splits the probe's long request into 2048 + 1812 tokens;
        # the second chunk reads the state the first one wrote.
        long_request, short_requests = 3860, [40] * 5
        first_chunk = 2048
        slots = list(range(1, len(short_requests) + 2))
        for state_dtype in (torch.bfloat16, torch.float32):
            for seed in SEEDS:
                with self.subTest(state_dtype=state_dtype, seed=seed):
                    generator = _generator(seed)
                    layer = _conv_layer(CHANNELS, generator)
                    lengths = [long_request] + short_requests
                    x = _randn((sum(lengths), CHANNELS), torch.bfloat16, generator)
                    state = _randn(
                        (len(slots) + 1, CHANNELS, STATE_LEN), state_dtype, generator
                    )

                    one_shot_state = state.clone()
                    one_shot, _ = _short_conv(
                        layer,
                        x,
                        _forward_batch(lengths, slots),
                        one_shot_state,
                        packed=False,
                    )

                    chunked_state = state.clone()
                    head, head_calls = _short_conv(
                        layer,
                        x[:first_chunk].contiguous(),
                        _forward_batch([first_chunk], slots[:1]),
                        chunked_state,
                        packed=True,
                    )
                    tail, tail_calls = _short_conv(
                        layer,
                        x[first_chunk:].contiguous(),
                        _forward_batch(
                            [long_request - first_chunk] + short_requests, slots
                        ),
                        chunked_state,
                        packed=True,
                    )

                    self.assertEqual((head_calls, tail_calls), (1, 1))
                    self.assert_bits_equal(
                        torch.cat([head, tail]), one_shot, "chunked output"
                    )
                    self.assert_bits_equal(
                        chunked_state[1:], one_shot_state[1:], "chunked state"
                    )


class _StubEmbedding:
    """Hands the PLE layer fresh embeddings, as the n-gram lookup would."""

    enable_ple_fusion = True

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def __call__(self, batch, forward_batch):
        return self._embeddings[: batch.processed_tokens].clone()


def _ple_layer(tokens, generator):
    """A Qwen4ExpPLELayer with the real norms, gate and conv at full width;
    only the n-gram lookup and the two projections are stand-ins."""
    layer = _conv_layer(CHANNELS, generator)
    layer.hidden_size = HIDDEN
    layer.hc_count = HC_COUNT
    layer._prefetch_state = None
    layer.ple_embedding = _StubEmbedding(
        _randn((tokens, HIDDEN), torch.bfloat16, generator)
    )
    key_weight = _randn((CHANNELS, HIDDEN), torch.bfloat16, generator) / 64
    value_weight = _randn((HIDDEN, HIDDEN), torch.bfloat16, generator) / 64
    layer.key_proj = lambda embeddings: (F.linear(embeddings, key_weight), None)
    layer.value_proj = lambda embeddings: (F.linear(embeddings, value_weight), None)
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = qwen4_exp.Qwen4ExpPLEGroupedNorm(CHANNELS, group_size=HIDDEN).to(
            device="cuda", dtype=torch.bfloat16
        )
        norm.weight.requires_grad_(False)
        setattr(layer, name, norm)
    return layer


def _peak_bytes(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del result
    return peak


@requires_cuda
class TestQwen4PackedPrefillMemory(CustomTestCase):
    # The probe that OOMed: one long request beside five short ones.
    LENGTHS = [3860] + [40] * 5
    BUCKET = 4096

    def setUp(self):
        self.tokens = sum(self.LENGTHS)
        self.slots = list(range(1, len(self.LENGTHS) + 1))
        generator = _generator(0)
        self.state = _randn(
            (len(self.slots) + 1, CHANNELS, STATE_LEN), torch.bfloat16, generator
        )
        self.generator = generator

    def _forward_batch(self):
        return _forward_batch(self.LENGTHS, self.slots, processed_tokens=self.tokens)

    def test_short_conv_peak_scales_with_tokens(self):
        layer = _conv_layer(CHANNELS, self.generator)
        x = _randn((self.tokens, CHANNELS), torch.bfloat16, self.generator)
        peaks = {}
        for packed in (True, False):
            state = self.state.clone()
            with _Path(state, packed=packed) as path:
                forward_batch = self._forward_batch()
                batch = _prepare(forward_batch, self.tokens)
                layer._short_conv(x, forward_batch, batch)  # warm up and compile
                peaks[packed] = _peak_bytes(
                    partial(layer._short_conv, x, forward_batch, batch)
                )
                self.assertEqual(path.packed_calls.call_count, 2 if packed else 0)

        bound = 2 * self.tokens * CHANNELS * 2 + 8 * MiB
        rows, longest = len(self.LENGTHS), max(self.LENGTHS)
        print(
            f"_short_conv peak at {self.LENGTHS[0]}+{self.LENGTHS[1:]}: "
            f"packed {peaks[True] / MiB:.1f} MiB (bound {bound / MiB:.1f}), "
            f"padded {peaks[False] / MiB:.1f} MiB"
        )
        self.assertLessEqual(peaks[True], bound)
        # The bug itself: padded_seq, conv_input and the conv output are
        # [requests, longest, channels] each and alive together.
        self.assertGreater(peaks[False], 3 * rows * longest * CHANNELS * 2)

    def test_ple_forward_peak_with_early_frees(self):
        layer = _ple_layer(self.tokens, self.generator)
        hidden_states = _randn((self.BUCKET, CHANNELS), torch.bfloat16, self.generator)
        peaks = {}
        for packed in (True, False):
            state = self.state.clone()
            with _Path(state, packed=packed):
                forward_batch = self._forward_batch()
                batch = _prepare(forward_batch, self.BUCKET)
                self.assertIsNotNone(batch.query_start_loc)
                layer(hidden_states, forward_batch, batch)  # warm up and compile
                peaks[packed] = _peak_bytes(
                    partial(layer, hidden_states, forward_batch, batch)
                )

        print(
            f"PLE forward peak at {self.LENGTHS[0]}+{self.LENGTHS[1:]} "
            f"(bucket {self.BUCKET}): packed {peaks[True] / MiB:.1f} MiB, "
            f"padded {peaks[False] / MiB:.1f} MiB"
        )
        self.assertLessEqual(peaks[True], 420 * MiB)
        self.assertGreater(peaks[False], peaks[True])


if __name__ == "__main__":
    unittest.main()
