"""Logprob normalization in the batch result processor must be idempotent.

The logprob fields reach the scheduler as tensors on some paths and as plain
Python lists on others (managers/utils.py converts with
``_async_d2h(v) if torch.is_tensor(v) else v``, and LogitsProcessorOutput types
next_token_token_ids_logprobs_val as ``List[Union[List[float], Tensor]]``). A
bare ``.tolist()`` on an already-converted value raised AttributeError inside
the scheduler and took the whole engine down.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


TOP_VAL = [[-0.1, -2.5], [-0.3, -1.7]]
TOP_IDX = [[5, 9], [7, 2]]
TOKEN_IDS_VAL = [[-4.0, -6.0], [-3.5]]
NEXT = [-0.1, -0.3]
INPUT = [-1.0, -2.0, -3.0]


def _output(as_tensors: bool) -> LogitsProcessorOutput:
    def wrap(rows):
        return (
            [torch.tensor(r) for r in rows] if as_tensors else [list(r) for r in rows]
        )

    return LogitsProcessorOutput(
        next_token_logits=None,
        next_token_logprobs=torch.tensor(NEXT) if as_tensors else list(NEXT),
        next_token_top_logprobs_val=wrap(TOP_VAL),
        next_token_top_logprobs_idx=wrap(TOP_IDX),
        next_token_token_ids_logprobs_val=wrap(TOKEN_IDS_VAL),
        next_token_token_ids_logprobs_idx=[[1, 2], [3]],
        input_token_logprobs=torch.tensor(INPUT) if as_tensors else list(INPUT),
    )


def _batch():
    return SimpleNamespace(
        return_logprob=True,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
    )


def _assert_rows_close(test, got, expected):
    test.assertEqual(len(got), len(expected))
    for got_row, expected_row in zip(got, expected):
        test.assertIsInstance(got_row, list)
        test.assertEqual(len(got_row), len(expected_row))
        for a, b in zip(got_row, expected_row):
            test.assertAlmostEqual(a, b, places=5)


class TestPrefillLogprobsToCpu(CustomTestCase):
    """move_logprobs_to_cpu (prefill / extend results)."""

    def _run(self, logits_output):
        SchedulerBatchResultProcessor.move_logprobs_to_cpu(
            MagicMock(), batch=_batch(), logits_output=logits_output
        )
        return logits_output

    def _check(self, out):
        self.assertIsInstance(out.next_token_logprobs, list)
        for a, b in zip(out.next_token_logprobs, NEXT):
            self.assertAlmostEqual(a, b, places=5)
        self.assertIsInstance(out.input_token_logprobs, tuple)
        for a, b in zip(out.input_token_logprobs, INPUT):
            self.assertAlmostEqual(a, b, places=5)
        _assert_rows_close(self, out.next_token_top_logprobs_val, TOP_VAL)
        self.assertEqual(out.next_token_top_logprobs_idx, TOP_IDX)
        _assert_rows_close(self, out.next_token_token_ids_logprobs_val, TOKEN_IDS_VAL)

    def test_tensors_are_converted(self):
        self._check(self._run(_output(as_tensors=True)))

    def test_already_converted_values_pass_through(self):
        self._check(self._run(_output(as_tensors=False)))

    def test_conversion_is_idempotent(self):
        self._check(self._run(self._run(_output(as_tensors=True))))


class TestDecodeLogprobsNormalization(CustomTestCase):
    """_normalize_decode_outputs (the site of the 2026-09-21 crash)."""

    def _run(self, logits_output):
        next_token_ids, next_token_logprobs = (
            SchedulerBatchResultProcessor._normalize_decode_outputs(
                MagicMock(),
                batch=_batch(),
                result=None,
                logits_output=logits_output,
                next_token_ids=torch.tensor([11, 12]),
            )
        )
        self.assertEqual(next_token_ids, [[11], [12]])
        return logits_output, next_token_logprobs

    def _check(self, out, next_token_logprobs):
        self.assertIsInstance(next_token_logprobs, list)
        for a, b in zip(next_token_logprobs, NEXT):
            self.assertAlmostEqual(a, b, places=5)
        _assert_rows_close(self, out.next_token_top_logprobs_val, TOP_VAL)
        self.assertEqual(out.next_token_top_logprobs_idx, TOP_IDX)
        _assert_rows_close(self, out.next_token_token_ids_logprobs_val, TOKEN_IDS_VAL)

    def test_tensors_are_converted(self):
        self._check(*self._run(_output(as_tensors=True)))

    def test_already_converted_values_pass_through(self):
        self._check(*self._run(_output(as_tensors=False)))

    def test_mixed_tensor_and_list_rows(self):
        """One row still a tensor, the other already a list."""
        out = _output(as_tensors=True)
        out.next_token_token_ids_logprobs_val[1] = list(TOKEN_IDS_VAL[1])
        out.next_token_top_logprobs_val[0] = list(TOP_VAL[0])
        out.next_token_top_logprobs_idx[0] = list(TOP_IDX[0])
        self._check(*self._run(out))

    def test_second_pass_over_the_same_output(self):
        """The decode normalizer rewrites the row lists in place but leaves
        next_token_logprobs a tensor, so a second pass gets past that line and
        used to die on the first row list -- the exact traceback observed:
        ``v.tolist() for v in logits_output.next_token_token_ids_logprobs_val``.
        """
        out, _ = self._run(_output(as_tensors=True))
        self._check(*self._run(out))


if __name__ == "__main__":
    unittest.main()
