"""CPU tests for Flash-Next (Qwen4-Exp) SM120 online FP8 wiring.

Covers which projections a Flash-Next decoder layer converts to MXFP8, the
checked MXFP8 method it installs, and the post-load rowwise-FP8 output head.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    configure_online_fp8,
    replace_linear_weight_rowwise_fp8,
    rowwise_scale_of,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpLayerExtensionMixin,
    Qwen4ExpPLELayer,
)
from sglang.test.test_utils import CustomTestCase

HIDDEN = 2560


def _linear(rows, columns, *, dtype=torch.bfloat16, quant_method=True):
    """A LinearBase stand-in: a meta weight and an unquantized method."""
    module = nn.Module()
    module.weight = nn.Parameter(
        torch.empty(rows, columns, dtype=dtype, device="meta"), requires_grad=False
    )
    if quant_method:
        module.quant_method = UnquantizedLinearMethod()
    return module


def _bare(cls):
    """An instance of an excluded module type without running its __init__."""
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


def _moe_block():
    mlp = nn.Module()
    mlp.gate = _linear(512, HIDDEN)  # router: excluded by name
    mlp.experts = _bare(FusedMoE)  # NVFP4 experts: excluded by type
    mlp.experts.inner = _linear(640, HIDDEN)
    mlp.shared_expert = nn.Module()
    mlp.shared_expert.gate_up_proj = _linear(1280, HIDDEN)
    mlp.shared_expert.down_proj = _linear(HIDDEN, 640)
    mlp.shared_expert_gate = _linear(1, HIDDEN)  # N < 128
    return mlp


def _hyper_connection():
    hc = nn.Module()
    # GatedResidual's mix weights are plain nn.Linear: rowwise FP8, not MXFP8.
    hc.input_mix_weight_down = nn.Linear(4 * HIDDEN, 320, bias=False, device="meta")
    hc.block_inject_weight = nn.Linear(4 * HIDDEN, 4, bias=False, device="meta")
    return hc


def _gdn_layer():
    layer = nn.Module()
    layer.linear_attn = nn.Module()
    layer.linear_attn.in_proj_qkvz = _linear(16384, HIDDEN)
    layer.linear_attn.in_proj_ba = _linear(96, HIDDEN)  # N < 128
    layer.linear_attn.conv1d = _linear(10240, 4)  # K < 128
    layer.linear_attn.out_proj = _linear(HIDDEN, 6144)
    layer.mlp = _moe_block()
    layer.attn_hyper_connection = _hyper_connection()
    layer.ple = _bare(Qwen4ExpPLELayer)  # PLE path: excluded by type
    layer.ple.key_proj = _linear(4 * HIDDEN, HIDDEN)
    layer.ple.value_proj = _linear(HIDDEN, HIDDEN)
    return layer


def _attention_layer():
    layer = nn.Module()
    layer.qkv_proj = _linear(13312, HIDDEN)
    layer.o_proj = _linear(HIDDEN, 6144)
    layer.indexer = nn.Module()
    layer.indexer.index_qk_proj = _linear(640, HIDDEN)
    layer.mlp = _moe_block()
    layer.mlp_hyper_connection = _hyper_connection()
    return layer


def _patched_mxfp8_backend(backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS):
    return mock.patch.multiple(
        fp8_module,
        resolve_mxfp8_dense_gemm_backend=mock.Mock(return_value=backend),
        dispatch_w8a8_mxfp8_linear=mock.Mock(return_value=lambda **kwargs: None),
    )


class TestQwen4Mxfp8Eligibility(CustomTestCase):
    def tearDown(self):
        configure_online_fp8(False, cuda_available=False, capability=None)

    def _convert(self, layer):
        Qwen4ExpLayerExtensionMixin._maybe_convert_linears_to_mxfp8(layer)
        return layer

    def test_option_off_leaves_every_projection_unquantized(self):
        layer = self._convert(_gdn_layer())
        self.assertFalse(hasattr(layer, "_online_mxfp8_linears"))
        self.assertIsInstance(
            layer.linear_attn.in_proj_qkvz.quant_method, UnquantizedLinearMethod
        )

    def test_gdn_layer_converts_exactly_its_large_projections(self):
        configure_online_fp8(True, cuda_available=True, capability=(12, 0))
        with _patched_mxfp8_backend():
            layer = self._convert(_gdn_layer())
        self.assertEqual(
            layer._online_mxfp8_linears,
            [
                "linear_attn.in_proj_qkvz",
                "linear_attn.out_proj",
                "mlp.shared_expert.gate_up_proj",
                "mlp.shared_expert.down_proj",
            ],
        )
        method = layer.linear_attn.in_proj_qkvz.quant_method
        self.assertEqual(type(method).__name__, "CheckedOnlineMxfp8LinearMethod")
        self.assertIsInstance(method, fp8_module.Fp8LinearMethod)
        self.assertTrue(method.use_mxfp8)
        self.assertTrue(method.block_quant)
        self.assertFalse(method.is_checkpoint_fp8_serialized)
        self.assertIs(
            method.mxfp8_dense_backend, Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS
        )
        # One method instance is shared across the layer's projections.
        self.assertIs(layer.mlp.shared_expert.down_proj.quant_method, method)
        for unconverted in (
            layer.linear_attn.in_proj_ba,
            layer.linear_attn.conv1d,
            layer.mlp.gate,
            layer.mlp.experts.inner,
            layer.mlp.shared_expert_gate,
            layer.ple.key_proj,
            layer.ple.value_proj,
        ):
            self.assertIsInstance(unconverted.quant_method, UnquantizedLinearMethod)

    def test_attention_layer_converts_qkv_o_and_qsa_indexer(self):
        configure_online_fp8(True, cuda_available=True, capability=(12, 0))
        with _patched_mxfp8_backend():
            layer = self._convert(_attention_layer())
        self.assertEqual(
            layer._online_mxfp8_linears,
            [
                "qkv_proj",
                "o_proj",
                "indexer.index_qk_proj",
                "mlp.shared_expert.gate_up_proj",
                "mlp.shared_expert.down_proj",
            ],
        )

    def test_missing_mxfp8_kernel_fails_startup(self):
        configure_online_fp8(True, cuda_available=True, capability=(12, 0))
        with _patched_mxfp8_backend(Mxfp8DenseGemmBackend.UNSUPPORTED):
            with self.assertRaisesRegex(RuntimeError, "no MXFP8 dense kernel"):
                self._convert(_gdn_layer())

    def test_checked_method_rejects_an_invalid_post_load_state(self):
        configure_online_fp8(True, cuda_available=True, capability=(12, 0))
        with _patched_mxfp8_backend():
            layer = self._convert(_gdn_layer())
        method = layer.linear_attn.in_proj_qkvz.quant_method

        def good_state(self, module):
            module.weight = nn.Parameter(
                torch.zeros(256, 128, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            scale = nn.Parameter(
                torch.zeros(256, 4, dtype=torch.uint8), requires_grad=False
            )
            scale.format_ue8m0 = True
            module.weight_scale_inv = scale

        def bf16_left_behind(self, module):
            module.weight = nn.Parameter(
                torch.zeros(256, 128, dtype=torch.bfloat16), requires_grad=False
            )

        module = nn.Module()
        with mock.patch.object(
            fp8_module.Fp8LinearMethod, "process_weights_after_loading", good_state
        ):
            method.process_weights_after_loading(module)
        self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)

        with mock.patch.object(
            fp8_module.Fp8LinearMethod,
            "process_weights_after_loading",
            bf16_left_behind,
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid weight/scale state"):
                method.process_weights_after_loading(nn.Module())


def _fake_model(*, tie=False, modules=()):
    return SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=tie),
        pp_group=SimpleNamespace(is_last_rank=True),
        lm_head=nn.Linear(4, 8, bias=False, dtype=torch.bfloat16),
        modules=lambda: list(modules),
    )


class TestQwen4PostLoadHead(CustomTestCase):
    def tearDown(self):
        configure_online_fp8(False, cuda_available=False, capability=None)

    def test_post_load_is_option_off_noop(self):
        model = _fake_model()
        original = model.lm_head.weight

        Qwen4ExpForConditionalGeneration.post_load_weights(model)

        self.assertIs(model.lm_head.weight, original)
        self.assertEqual(model.lm_head.weight.dtype, torch.bfloat16)

    def test_post_load_replaces_head_once_and_rejects_tied_embeddings(self):
        configure_online_fp8(True, cuda_available=True, capability=(12, 0))
        model = _fake_model()
        Qwen4ExpForConditionalGeneration.post_load_weights(model)
        first = model.lm_head.weight
        self.assertEqual(first.dtype, torch.float8_e4m3fn)
        self.assertIsNotNone(rowwise_scale_of(first))

        Qwen4ExpForConditionalGeneration.post_load_weights(model)
        self.assertIs(model.lm_head.weight, first)

        model.config.tie_word_embeddings = True
        with self.assertRaisesRegex(RuntimeError, "tied"):
            Qwen4ExpForConditionalGeneration.post_load_weights(model)

    def test_post_load_requires_rowwise_hyperconnection_weights(self):
        configure_online_fp8(True, cuda_available=True, capability=(12, 0))

        def hc(converted):
            module = nn.Module()
            module._online_fp8_mix = True
            for name in ("input_mix_weight_down", "input_mix_weight_up"):
                linear = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
                if converted:
                    replace_linear_weight_rowwise_fp8(linear)
                setattr(module, name, linear)
            return module

        Qwen4ExpForConditionalGeneration.post_load_weights(
            _fake_model(modules=[hc(True), hc(True)])
        )
        with self.assertRaisesRegex(RuntimeError, "missing its rowwise scale"):
            Qwen4ExpForConditionalGeneration.post_load_weights(
                _fake_model(modules=[hc(True), hc(False)])
            )


if __name__ == "__main__":
    unittest.main()
