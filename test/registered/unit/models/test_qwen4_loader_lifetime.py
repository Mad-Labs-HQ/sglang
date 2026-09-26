"""Exercise the real Qwen4-Exp loader body with CPU-only lifetime/PLE fixtures.

AST extraction avoids initializing the GPU model; it does not rewrite the
method under test. This establishes reference lifetime, not VRAM savings.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import ast
import gc
import logging
import unittest
import weakref
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase

_LOGGER_NAME = "qwen4-loader-lifetime-test"


class _Parameter:
    pass


class _OtherModule:
    pass


class _PLE:
    def __init__(self):
        self.ngram_embedding = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.zeros((4, 2), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
            org_vocab_size=4,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=0, org_vocab_end_index=4
            ),
        )


class _Model:
    config = SimpleNamespace(
        num_experts=None,
        split_ngram_parts=2,
        tie_word_embeddings=False,
        encoder_only=False,
    )
    language_model_only = False
    pp_group = SimpleNamespace(is_last_rank=True)
    start_layer = 0
    end_layer = 1

    def __init__(self, ple=None):
        self.weight = _Parameter()
        self.ple = ple

    def named_parameters(self, **kwargs):
        return [("lm_head.weight", self.weight)]

    def named_buffers(self):
        return []

    def named_modules(self):
        return [("model.ple", self.ple)] if self.ple is not None else []

    def modules(self):
        return []

    def _load_qwen4_exp_ple_buffer(self, *args):
        return False

    def post_load_weights(self):
        # Precision conversion is tested independently; this fixture measures
        # whether the real loader releases its captured parameter snapshot.
        pass


def _real_loader():
    source = (
        Path(__file__).resolve().parents[4] / "python/sglang/srt/models/qwen4_exp.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Qwen4ExpForConditionalGeneration"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
    )
    namespace = {
        "torch": torch,
        "Iterable": Iterable,
        "Tuple": tuple,
        "Set": set,
        "_use_aiter": False,
        "FusedMoE": _OtherModule,
        "get_layer_id": lambda name: None,
        "default_weight_loader": None,
        "Qwen4ExpNGramEmbedding": _PLE,
        "Qwen4ExpPinnedHostEmbedding": _OtherModule,
        "Qwen3_5GatedDeltaNet": _OtherModule,
        "logger": logging.getLogger(_LOGGER_NAME),
    }
    exec(  # noqa: S102 -- execute only the repository's method under test
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["load_weights"]


class TestQwen4LoaderLifetime(CustomTestCase):
    def test_replaced_parameter_released_without_cyclic_gc(self):
        load = _real_loader()
        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        try:
            model = _Model()
            original = weakref.ref(model.weight)

            def weights():
                # The real loader already took its named-parameter snapshot when
                # iteration starts. Mimic a later parameter replacement during
                # load (the SM120 online-FP8 lm_head swap does exactly this).
                model.weight = _Parameter()
                yield from ()

            self.assertEqual(load(model, weights()), set())
            self.assertIsNone(
                original(), "loader retains replaced parameter until cyclic GC"
            )
        finally:
            gc.collect()
            if enabled:
                gc.enable()

    def test_ple_downcast_warns_once_per_load_and_copies_both_shards(self):
        load = _real_loader()
        model = _Model(_PLE())
        weights = [
            (
                f"model.ple.ngram_embedding.shard_{i}.weight",
                torch.full((2, 2), i + 1, dtype=torch.bfloat16),
            )
            for i in range(2)
        ]
        with self.assertLogs(_LOGGER_NAME, level=logging.WARNING) as logs:
            for _ in range(2):
                self.assertEqual(
                    load(model, weights), {"model.ple.ngram_embedding.weight"}
                )
        warnings = [line for line in logs.output if "downcasting is lossy" in line]
        self.assertEqual(len(warnings), 2)
        torch.testing.assert_close(
            model.ple.ngram_embedding.weight.float(),
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]),
        )


if __name__ == "__main__":
    unittest.main()
