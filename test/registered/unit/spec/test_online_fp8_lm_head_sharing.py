"""CPU tests: an SM120 online-FP8 rowwise lm_head through logits and FR-Spec."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.kernels.ops.gemm import sm120_online_fp8
from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    dequantize_rowwise_weight,
    replace_linear_weight_rowwise_fp8,
    rowwise_scale_of,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.test_utils import CustomTestCase


class _DraftModel:
    hot_token_id = None

    def __init__(self):
        self.installed = None

    def set_embed_and_head(self, embed, head):
        self.installed = (embed, head)


class _NonEagle3:
    @staticmethod
    def is_eagle3():
        return False


def _rowwise_head(rows=6, columns=4):
    head = nn.Linear(columns, rows, bias=False, dtype=torch.bfloat16)
    head.weight.data.copy_(torch.arange(rows * columns).reshape(rows, columns))
    head.weight.weight_loader = object()
    replace_linear_weight_rowwise_fp8(head)
    return head


def _eagle_worker(target_head, embed, hot_token_id):
    draft_model = _DraftModel()
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    lm_head=target_head,
                    get_embed_and_head=lambda: (embed, target_head.weight),
                )
            )
        ),
        draft_runner=SimpleNamespace(model=draft_model),
        hot_token_id=hot_token_id,
        speculative_algorithm=_NonEagle3(),
    )
    return worker, draft_model


class TestOnlineFp8LmHeadSharing(CustomTestCase):
    def test_eagle_hot_vocab_selects_weight_and_scale_rows_together(self):
        target_head = _rowwise_head()
        embed = nn.Parameter(torch.ones(6, 4))
        worker, draft_model = _eagle_worker(
            target_head, embed, hot_token_id=torch.tensor([5, 2, 0])
        )

        EagleDraftWorker.init_lm_head(worker)

        installed = draft_model.installed[1]
        self.assertIs(draft_model.installed[0], embed)
        self.assertIsInstance(installed, nn.Parameter)
        self.assertEqual(installed.dtype, torch.float8_e4m3fn)
        self.assertFalse(hasattr(installed, "weight_loader"))
        torch.testing.assert_close(
            rowwise_scale_of(installed),
            rowwise_scale_of(target_head.weight)[[5, 2, 0]],
        )
        torch.testing.assert_close(
            dequantize_rowwise_weight(installed),
            dequantize_rowwise_weight(target_head.weight)[[5, 2, 0]],
        )
        # The target's own head is untouched.
        self.assertEqual(target_head.weight.shape[0], 6)

    def test_eagle_hot_vocab_on_a_bf16_head_is_unchanged(self):
        target_head = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
        target_head.weight.data.copy_(torch.arange(24).reshape(6, 4))
        embed = nn.Parameter(torch.ones(6, 4))
        worker, draft_model = _eagle_worker(
            target_head, embed, hot_token_id=torch.tensor([5, 2, 0])
        )

        EagleDraftWorker.init_lm_head(worker)

        installed = draft_model.installed[1]
        self.assertEqual(installed.dtype, torch.bfloat16)
        self.assertIsNone(rowwise_scale_of(installed))
        torch.testing.assert_close(installed, target_head.weight[[5, 2, 0]])

    def test_qwen_mtp_target_module_sharing_retains_rowwise_metadata(self):
        target_head = _rowwise_head()
        draft = SimpleNamespace(
            config=SimpleNamespace(tie_word_embeddings=False),
            lm_head=nn.Linear(4, 6, bias=False),
        )

        Qwen3_5ForCausalLMMTP.set_lm_head_from_target(draft, target_head)

        self.assertIs(draft.lm_head, target_head)
        self.assertIs(
            rowwise_scale_of(draft.lm_head.weight),
            rowwise_scale_of(target_head.weight),
        )


class TestLogitsProcessorRowwiseHead(CustomTestCase):
    def test_rowwise_metadata_takes_priority_over_stale_quant_method(self):
        linear = _rowwise_head(rows=5)
        expected = torch.randn(2, 5)
        calls = []

        def rowwise_logits(hidden_states, weight):
            calls.append((hidden_states, weight))
            return expected

        class _StaleQuantMethod:
            def apply(self, *args, **kwargs):
                raise AssertionError("stale draft quant method must not run")

        processor = SimpleNamespace(use_fp32_lm_head=False, rl_on_policy_target=None)
        lm_head = SimpleNamespace(
            weight=linear.weight, quant_method=_StaleQuantMethod()
        )
        hidden = torch.randn(2, 4)

        with mock.patch.object(
            sm120_online_fp8, "rowwise_fp8_lm_head_logits", rowwise_logits
        ):
            actual = LogitsProcessor._compute_lm_head(processor, hidden, lm_head)

        self.assertIs(actual, expected)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], hidden)
        self.assertIs(calls[0][1], linear.weight)

    def test_rowwise_head_rejects_fp32_lm_head(self):
        linear = _rowwise_head(rows=5)
        processor = SimpleNamespace(use_fp32_lm_head=True, rl_on_policy_target=None)
        lm_head = SimpleNamespace(weight=linear.weight, quant_method=None)
        with self.assertRaisesRegex(RuntimeError, "use-fp32-lm-head"):
            LogitsProcessor._compute_lm_head(processor, torch.randn(2, 4), lm_head)

    def test_bf16_head_keeps_the_dense_matmul(self):
        weight = torch.randn(5, 4).bfloat16()
        processor = SimpleNamespace(use_fp32_lm_head=False, rl_on_policy_target=None)
        lm_head = SimpleNamespace(weight=weight, quant_method=None)
        hidden = torch.randn(2, 4).bfloat16()

        def must_not_run(*args, **kwargs):
            raise AssertionError("BF16 heads must not take the rowwise path")

        with mock.patch.object(
            sm120_online_fp8, "rowwise_fp8_lm_head_logits", must_not_run
        ):
            actual = LogitsProcessor._compute_lm_head(processor, hidden, lm_head)
        torch.testing.assert_close(actual, hidden @ weight.T)


if __name__ == "__main__":
    unittest.main()
