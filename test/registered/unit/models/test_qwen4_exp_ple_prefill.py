"""Qwen4 PLE prefill guards that need no GPU: when the packed short conv may
run, the warning when it cannot, and the valid-token mask it lets us skip."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen4_exp
from sglang.srt.utils import common as utils_common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PACKED_MODES = (ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL)


def _forward_batch(
    lengths, *, processed_tokens=None, host_lengths=True, mode=ForwardMode.EXTEND
):
    return SimpleNamespace(
        forward_mode=mode,
        _original_forward_mode=None,
        tbo_parent_token_range=None,
        spec_algorithm=None,
        spec_info=None,
        global_num_token_non_padded_cpu=processed_tokens,
        extend_seq_lens=torch.tensor(lengths, dtype=torch.int32),
        extend_seq_lens_cpu=list(lengths) if host_lengths else None,
        req_pool_indices=torch.arange(1, len(lengths) + 1),
        out_cache_loc=None,
        mamba_track_indices=None,
        mamba_track_mask=None,
    )


def _pool(conv_state=None):
    return SimpleNamespace(
        ple_window_cache=None,
        get_mamba_indices=lambda indices: indices,
        short_conv_layer_cache=lambda layer_id: conv_state,
    )


def _prepare(forward_batch, physical_tokens):
    with patch.object(qwen4_exp, "get_req_to_token_pool", return_value=_pool()):
        return qwen4_exp._prepare_ple_batch(
            torch.zeros(physical_tokens, dtype=torch.int64),
            forward_batch,
            ngram_size=None,
            ngram_eos_token_id=None,
        )


def _dispatch_args(tokens=4, channels=8, rows=2):
    """CPU stand-ins for _use_qwen4_varlen_prefill after the layout argument."""
    return (
        torch.zeros((tokens, channels), dtype=torch.bfloat16),
        torch.zeros((channels, 1, 4), dtype=torch.bfloat16),
        torch.zeros((rows + 1, channels, 9), dtype=torch.bfloat16),
        torch.arange(1, rows + 1),
        torch.zeros(tokens, dtype=torch.int64),
        torch.arange(tokens),
        3,
    )


class TestQwen4PackedPrefillLayout(CustomTestCase):
    def test_query_start_loc_needs_host_lengths_covering_processed_tokens(self):
        lengths = [3, 0, 2, 3]
        for mode in PACKED_MODES:
            with self.subTest(mode=mode):
                batch = _prepare(_forward_batch(lengths, mode=mode), 8)
                self.assertEqual(batch.query_start_loc.tolist(), [0, 3, 3, 5, 8])
        # Graph bucket padding beyond the processed tokens keeps the layout.
        batch = _prepare(_forward_batch(lengths, processed_tokens=8), 16)
        self.assertEqual(batch.query_start_loc.tolist(), [0, 3, 3, 5, 8])

        for name, forward_batch in (
            ("fewer processed tokens", _forward_batch(lengths, processed_tokens=7)),
            (
                "no host lengths",
                _forward_batch(lengths, processed_tokens=8, host_lengths=False),
            ),
        ):
            with self.subTest(name):
                batch = _prepare(forward_batch, 8)
                self.assertIsNone(batch.query_start_loc)
                with (
                    patch.object(qwen4_exp, "is_cuda", return_value=True),
                    patch.object(qwen4_exp, "print_warning_once"),
                ):
                    self.assertFalse(
                        qwen4_exp._use_qwen4_varlen_prefill(
                            batch.mode, batch.query_start_loc, *_dispatch_args()
                        )
                    )

    def test_replay_rebuilds_the_packed_layout_from_the_live_batch(self):
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            breakable_cuda_graph as bcg,
        )

        context = SimpleNamespace(forward_batch=_forward_batch([8]))
        capture = Mock(_barrier_fn=None, cuda_graph=SimpleNamespace(_break_fns=[]))
        with (
            patch.object(qwen4_exp, "get_req_to_token_pool", return_value=_pool()),
            patch.object(bcg, "_current_capture_var") as active,
            patch.object(
                qwen4_exp, "get_tc_piecewise_forward_context", return_value=context
            ),
        ):
            active.get.return_value = capture
            holder = qwen4_exp._breakable_prepare_ple_batch(
                input_ids=torch.arange(8), ngram_size=None, ngram_eos_token_id=None
            )
            self.assertEqual(holder.batch.query_start_loc.tolist(), [0, 8])
            # The 8-token bucket replays live batches, including one whose
            # processed count disagrees with its host lengths.
            for forward_batch, expected in (
                (_forward_batch([3, 0, 2, 3]), [0, 3, 3, 5, 8]),
                (_forward_batch([3, 0, 2, 3], processed_tokens=7), None),
                (_forward_batch([2, 3], processed_tokens=5), [0, 2, 5]),
            ):
                context.forward_batch = forward_batch
                for replay in capture.cuda_graph._break_fns:
                    replay()
                actual = holder.batch.query_start_loc
                self.assertEqual(None if actual is None else actual.tolist(), expected)

    def test_packed_layout_marks_every_processed_token_valid(self):
        # Skipping the valid-token torch.where is exact only if this holds.
        generator = torch.Generator().manual_seed(0)
        for trial in range(200):
            rows = int(torch.randint(1, 9, (), generator=generator))
            lengths = torch.randint(0, 12, (rows,), generator=generator)
            lengths[torch.rand(rows, generator=generator) < 0.3] = 0
            lengths = lengths.tolist()
            processed = sum(lengths)
            padding = int(torch.randint(0, 5, (), generator=generator))
            with self.subTest(trial=trial, lengths=lengths, padding=padding):
                batch = _prepare(
                    _forward_batch(lengths, processed_tokens=processed),
                    processed + padding,
                )
                self.assertIsNotNone(batch.query_start_loc)
                self.assertTrue(bool(batch.valid_tokens.all()))


class TestQwen4PaddedPrefillWarning(CustomTestCase):
    def setUp(self):
        qwen4_exp.print_warning_once.cache_clear()
        self.addCleanup(qwen4_exp.print_warning_once.cache_clear)

    def _fallback_warnings(self, records):
        return [
            r.getMessage() for r in records if "padded short conv" in r.getMessage()
        ]

    def test_each_cause_warns_once(self):
        args = _dispatch_args()
        query_start_loc = torch.tensor([0, 2, 4])
        with (
            patch.object(qwen4_exp, "is_cuda", return_value=True),
            self.assertLogs(utils_common.logger, level="WARNING") as logs,
        ):
            for _ in range(3):
                # Lengths that do not cover the processed tokens.
                self.assertFalse(
                    qwen4_exp._use_qwen4_varlen_prefill(ForwardMode.EXTEND, None, *args)
                )
                # Inputs the packed kernel does not take (here: CPU tensors).
                self.assertFalse(
                    qwen4_exp._use_qwen4_varlen_prefill(
                        ForwardMode.EXTEND, query_start_loc, *args
                    )
                )
                with patch.object(
                    qwen4_exp.envs.SGLANG_ENABLE_QWEN4_PLE_FUSION,
                    "get",
                    return_value=False,
                ):
                    self.assertFalse(
                        qwen4_exp._use_qwen4_varlen_prefill(
                            ForwardMode.EXTEND, query_start_loc, *args
                        )
                    )

        warnings = self._fallback_warnings(logs.records)
        self.assertEqual(len(warnings), 3, warnings)
        for cause in (
            "do not cover the processed tokens",
            "unsupported inputs: x torch.bfloat16 cpu",
            "SGLANG_ENABLE_QWEN4_PLE_FUSION is off",
        ):
            self.assertEqual(sum(cause in w for w in warnings), 1, (cause, warnings))

    def test_modes_without_a_packed_path_stay_quiet(self):
        args = _dispatch_args()
        with (
            patch.object(qwen4_exp, "is_cuda", return_value=True),
            patch.object(utils_common.logger, "warning") as warning,
        ):
            for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                self.assertFalse(qwen4_exp._use_qwen4_varlen_prefill(mode, None, *args))
        with (
            patch.object(qwen4_exp, "is_cuda", return_value=False),
            patch.object(utils_common.logger, "warning") as off_cuda,
        ):
            self.assertFalse(
                qwen4_exp._use_qwen4_varlen_prefill(ForwardMode.EXTEND, None, *args)
            )
        warning.assert_not_called()
        off_cuda.assert_not_called()


class _StubEmbedding:
    enable_ple_fusion = True

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def __call__(self, batch, forward_batch):
        return self._embeddings[: batch.processed_tokens].clone()


def _ple_layer(tokens, generator, *, hidden=16, hc_count=2):
    channels = hidden * hc_count
    layer = qwen4_exp.Qwen4ExpPLELayer.__new__(qwen4_exp.Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.layer_id = 0
    layer.hidden_size = hidden
    layer.hc_count = hc_count
    layer.conv_channels = channels
    layer.short_conv_dilation = 3
    layer.short_conv_state_len = 9
    layer._prefetch_state = None
    layer.conv1d = nn.Conv1d(
        channels, channels, 4, groups=channels, dilation=3, padding=9, bias=False
    )
    layer.conv1d.weight.requires_grad_(False)
    layer.conv1d.weight.copy_(torch.randn(channels, 1, 4, generator=generator))
    layer.ple_embedding = _StubEmbedding(
        torch.randn(tokens, hidden, generator=generator)
    )
    key_weight = torch.randn(channels, hidden, generator=generator)
    value_weight = torch.randn(hidden, hidden, generator=generator)
    layer.key_proj = lambda embeddings: (F.linear(embeddings, key_weight), None)
    layer.value_proj = lambda embeddings: (F.linear(embeddings, value_weight), None)
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = qwen4_exp.Qwen4ExpPLEGroupedNorm(channels, group_size=hidden)
        norm.weight.requires_grad_(False)
        norm.weight.copy_(torch.randn(channels, generator=generator) / 4)
        setattr(layer, name, norm)
    return layer


class TestQwen4PackedPrefillSkipsTheMask(CustomTestCase):
    def test_skipping_the_mask_matches_torch_where_bitwise(self):
        generator = torch.Generator().manual_seed(0)
        for lengths, padding in (([5, 0, 12, 1], 3), ([7], 0), ([2, 3, 0], 1)):
            processed = sum(lengths)
            layer = _ple_layer(processed, generator)
            hidden_states = torch.randn(
                processed + padding,
                layer.conv_channels,
                generator=generator,
            )
            state = torch.randn(
                len(lengths) + 1, layer.conv_channels, 9, generator=generator
            )
            forward_batch = _forward_batch(lengths, processed_tokens=processed)
            batch = _prepare(forward_batch, processed + padding)
            self.assertIsNotNone(batch.query_start_loc)
            masked_batch = msgspec.structs.replace(batch, query_start_loc=None)

            outputs = []
            for ple_batch in (batch, masked_batch):
                with (
                    patch.object(qwen4_exp, "is_cuda", return_value=False),
                    patch.object(
                        qwen4_exp,
                        "get_req_to_token_pool",
                        return_value=_pool(state.clone()),
                    ),
                    torch.no_grad(),
                ):
                    outputs.append(layer(hidden_states, forward_batch, ple_batch))

            with self.subTest(lengths=lengths, padding=padding):
                skipped, masked = outputs
                self.assertEqual(skipped.shape, masked.shape)
                self.assertTrue(
                    torch.equal(skipped.view(torch.int32), masked.view(torch.int32))
                )


if __name__ == "__main__":
    unittest.main()
