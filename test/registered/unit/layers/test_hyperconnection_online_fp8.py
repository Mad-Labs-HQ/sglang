"""CPU tests for SM120 online-FP8 HyperConnection mix-weight placement."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    attach_rowwise_ingest,
    configure_online_fp8,
    dequantize_rowwise_weight,
    rowwise_scale_of,
)
from sglang.srt.layers.hc_mix_triton import _rowwise_fp8_pair, fused_hc_mix_supported
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.test.test_utils import CustomTestCase

HC_COUNT = 4
HIDDEN_SIZE = 64
HC_LOWRANK = 32


def _config():
    return HyperConnectionConfig(
        hc_count=HC_COUNT,
        hidden_size=HIDDEN_SIZE,
        params_dtype=torch.bfloat16,
        hc_lowrank=HC_LOWRANK,
        hc_per_branch_norm=True,
    )


def _cpu_device_module():
    return mock.patch.object(
        torch,
        "get_device_module",
        return_value=SimpleNamespace(current_device=lambda: "cpu"),
    )


class TestHyperConnectionOnlineFp8(CustomTestCase):
    def tearDown(self):
        configure_online_fp8(False, cuda_available=False, capability=None)

    def test_option_off_keeps_resident_bf16_mix_weights(self):
        with _cpu_device_module():
            layer = GatedResidual(
                _config(), use_mix=True, use_combine=False, online_fp8=True
            )
        self.assertFalse(layer._online_fp8_mix)
        for linear in (layer.input_mix_weight_down, layer.input_mix_weight_up):
            self.assertEqual(linear.weight.device.type, "cpu")
            self.assertEqual(linear.weight.dtype, torch.bfloat16)
            self.assertFalse(hasattr(linear.weight, "weight_loader"))

    def test_unflagged_module_ignores_the_switch(self):
        configure_online_fp8(True, cuda_available=True, capability=(12, 0))
        with _cpu_device_module():
            layer = GatedResidual(_config(), use_mix=True, use_combine=False)
        self.assertFalse(layer._online_fp8_mix)
        self.assertEqual(layer.input_mix_weight_down.weight.device.type, "cpu")

    def test_option_on_births_mix_weights_on_meta_with_quantizing_loaders(self):
        configure_online_fp8(True, cuda_available=True, capability=(12, 0))
        layer = GatedResidual(
            _config(), use_mix=True, use_combine=False, online_fp8=True
        )
        self.assertTrue(layer._online_fp8_mix)
        # The CuTe split-K pair reads BF16 mix weights; never route FP8 to it.
        self.assertFalse(layer._jit_mix_ok)
        down, up = layer.input_mix_weight_down, layer.input_mix_weight_up
        self.assertTrue(down.weight.is_meta)
        self.assertTrue(up.weight.is_meta)
        self.assertTrue(hasattr(down.weight, "weight_loader"))
        self.assertTrue(hasattr(up.weight, "weight_loader"))

        # Retarget the (still meta) weights to the host for this CPU test.
        attach_rowwise_ingest([down, up], target_device=torch.device("cpu"))
        generator = torch.Generator().manual_seed(0)
        down_bf16 = torch.randn(
            (HC_LOWRANK, HC_COUNT * HIDDEN_SIZE), generator=generator
        ).bfloat16()
        up_bf16 = torch.randn(
            (HC_COUNT * HIDDEN_SIZE, HC_LOWRANK), generator=generator
        ).bfloat16()
        down.weight.weight_loader(down.weight, down_bf16)
        up.weight.weight_loader(up.weight, up_bf16)

        for linear, reference in ((down, down_bf16), (up, up_bf16)):
            self.assertEqual(linear.weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(linear.weight.device.type, "cpu")
            self.assertEqual(
                tuple(rowwise_scale_of(linear.weight).shape),
                (reference.shape[0],),
            )
            torch.testing.assert_close(
                dequantize_rowwise_weight(linear.weight),
                reference,
                rtol=0.07,
                atol=0.07,
            )

        down_scale, up_scale = _rowwise_fp8_pair(down.weight, up.weight)
        self.assertIs(down_scale, rowwise_scale_of(down.weight))
        self.assertIs(up_scale, rowwise_scale_of(up.weight))
        # CPU tensors never take the CUDA fused kernel.
        hyper_input_normed = torch.zeros(
            (4, HC_COUNT * HIDDEN_SIZE), dtype=torch.bfloat16
        )
        self.assertFalse(
            fused_hc_mix_supported(hyper_input_normed, down.weight, up.weight)
        )

    def test_rowwise_pair_rejects_a_partial_conversion(self):
        bf16 = torch.zeros((8, 8), dtype=torch.bfloat16)
        self.assertIsNone(_rowwise_fp8_pair(bf16, bf16))

        fp8_without_scale = torch.zeros((8, 8), dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(RuntimeError, "complete rowwise-FP8"):
            _rowwise_fp8_pair(fp8_without_scale, bf16)
        with self.assertRaisesRegex(RuntimeError, "complete rowwise-FP8"):
            _rowwise_fp8_pair(fp8_without_scale, fp8_without_scale)


if __name__ == "__main__":
    unittest.main()
