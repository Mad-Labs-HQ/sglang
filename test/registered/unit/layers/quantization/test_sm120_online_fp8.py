"""CPU tests for the SM120 online-FP8 switch and its rowwise/MXFP8 helpers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    attach_rowwise_ingest,
    configure_online_fp8,
    convert_eligible_linears_to_mxfp8,
    dequantize_rowwise_weight,
    online_fp8_enabled,
    quantize_rowwise_fp8,
    replace_linear_weight_rowwise_fp8,
    rowwise_scale_of,
    select_rowwise_weight_rows,
)
from sglang.test.test_utils import CustomTestCase


class _Unquantized:
    pass


class _Excluded(nn.Module):
    pass


class _Linear(nn.Module):
    def __init__(self, rows=128, columns=128, *, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(rows, columns, dtype=dtype))
        self.quant_method = _Unquantized()


class TestOnlineFp8Switch(CustomTestCase):
    def tearDown(self):
        configure_online_fp8(False, cuda_available=False, capability=None)

    def test_online_fp8_is_opt_in_and_exact_sm120(self):
        self.assertIs(
            configure_online_fp8(False, cuda_available=False, capability=None), False
        )
        self.assertIs(online_fp8_enabled(), False)

        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            configure_online_fp8(True, cuda_available=False, capability=None)
        with self.assertRaisesRegex(RuntimeError, "exactly SM120"):
            configure_online_fp8(True, cuda_available=True, capability=(12, 1))
        with self.assertRaisesRegex(RuntimeError, "exactly SM120"):
            configure_online_fp8(True, cuda_available=True, capability=(10, 0))
        self.assertIs(online_fp8_enabled(), False)

        self.assertIs(
            configure_online_fp8(True, cuda_available=True, capability=(12, 0)), True
        )
        self.assertIs(online_fp8_enabled(), True)

    def test_environment_switch_defaults_off(self):
        from sglang.srt.environ import envs

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_SM120_ONLINE_MXFP8", None)
            self.assertIs(envs.SGLANG_SM120_ONLINE_MXFP8.get(), False)
        with mock.patch.dict(os.environ, {"SGLANG_SM120_ONLINE_MXFP8": "true"}):
            self.assertIs(envs.SGLANG_SM120_ONLINE_MXFP8.get(), True)
        with mock.patch.dict(os.environ, {"SGLANG_SM120_ONLINE_MXFP8": "false"}):
            self.assertIs(envs.SGLANG_SM120_ONLINE_MXFP8.get(), False)

    def test_bf16_gemm_config_resolves_the_switch_and_fails_off_sm120(self):
        from sglang.srt.layers.quantization import unquant

        exec_config = SimpleNamespace(
            kernel=SimpleNamespace(bf16_gemm_backend="auto"),
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
        )
        platform = SimpleNamespace(is_sm100=False)
        common = [
            mock.patch.object(unquant, "get_exec", return_value=exec_config),
            mock.patch.object(unquant, "get_platform", return_value=platform),
            mock.patch.object(
                unquant, "should_enable_bf16_splitk_gemm", return_value=False
            ),
        ]
        for patcher in common:
            patcher.start()
            self.addCleanup(patcher.stop)
        original_backend = unquant._BF16_GEMM_BACKEND
        self.addCleanup(setattr, unquant, "_BF16_GEMM_BACKEND", original_backend)

        with mock.patch.dict(os.environ, {"SGLANG_SM120_ONLINE_MXFP8": "false"}):
            unquant.initialize_bf16_gemm_config()
        self.assertIs(online_fp8_enabled(), False)

        with (
            mock.patch.dict(os.environ, {"SGLANG_SM120_ONLINE_MXFP8": "true"}),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(12, 0)
            ),
        ):
            unquant.initialize_bf16_gemm_config()
        self.assertIs(online_fp8_enabled(), True)

        # An explicit request on unsupported hardware stops startup.
        with mock.patch.dict(os.environ, {"SGLANG_SM120_ONLINE_MXFP8": "true"}):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                unquant.initialize_bf16_gemm_config()
        self.assertIs(online_fp8_enabled(), False)


class TestMxfp8Eligibility(CustomTestCase):
    def test_candidate_conversion_is_bounded_and_option_off_is_noop(self):
        root = nn.Module()
        root.proj = _Linear()
        root.small = _Linear(rows=96)
        root.narrow_k = _Linear(columns=96)
        root.unaligned_k = _Linear(columns=144)
        root.gate = _Linear()
        root.experts = _Excluded()
        root.experts.proj = _Linear()
        root.wrong_dtype = _Linear(dtype=torch.float32)
        root.nested = nn.Module()
        root.nested.proj = _Linear(rows=256, columns=512)

        calls = []

        def factory():
            method = SimpleNamespace(kind="mxfp8")
            calls.append(method)
            return method

        self.assertEqual(
            convert_eligible_linears_to_mxfp8(
                root,
                enabled=False,
                method_factory=factory,
                unquantized_method_type=_Unquantized,
                excluded_module_type=_Excluded,
            ),
            [],
        )
        self.assertEqual(calls, [])
        self.assertIsInstance(root.proj.quant_method, _Unquantized)

        converted = convert_eligible_linears_to_mxfp8(
            root,
            enabled=True,
            method_factory=factory,
            unquantized_method_type=_Unquantized,
            excluded_module_type=_Excluded,
        )
        self.assertEqual(converted, ["proj", "nested.proj"])
        self.assertEqual(len(calls), 1)
        self.assertIs(root.proj.quant_method, calls[0])
        self.assertIs(root.nested.proj.quant_method, calls[0])
        for name in (
            "small",
            "narrow_k",
            "unaligned_k",
            "gate",
            "wrong_dtype",
        ):
            self.assertIsInstance(getattr(root, name).quant_method, _Unquantized)
        self.assertIsInstance(root.experts.proj.quant_method, _Unquantized)

    def test_no_candidate_never_builds_a_method(self):
        root = nn.Module()
        root.small = _Linear(rows=64)

        def factory():
            raise AssertionError("factory must not run without candidates")

        self.assertEqual(
            convert_eligible_linears_to_mxfp8(
                root,
                enabled=True,
                method_factory=factory,
                unquantized_method_type=_Unquantized,
                excluded_module_type=_Excluded,
            ),
            [],
        )


class TestRowwiseFp8(CustomTestCase):
    def test_quantize_rejects_non_bf16_or_non_2d(self):
        with self.assertRaises(TypeError):
            quantize_rowwise_fp8(torch.ones(4, 4, dtype=torch.float32))
        with self.assertRaises(TypeError):
            quantize_rowwise_fp8(torch.ones(4, dtype=torch.bfloat16))

    def test_rowwise_quantization_round_trip_and_replacement_are_idempotent(self):
        weight = torch.tensor(
            [[-4.0, -1.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.bfloat16,
        )
        linear = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
        linear.weight.data.copy_(weight)
        linear.weight.weight_loader = object()
        linear.weight.output_dim = 0

        freed = replace_linear_weight_rowwise_fp8(linear)
        self.assertEqual(freed, weight.numel() * weight.element_size())
        self.assertEqual(linear.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(rowwise_scale_of(linear.weight).shape), (2,))
        self.assertEqual(rowwise_scale_of(linear.weight).dtype, torch.float32)
        self.assertTrue(hasattr(linear.weight, "weight_loader"))
        self.assertEqual(linear.weight.output_dim, 0)
        torch.testing.assert_close(
            dequantize_rowwise_weight(linear.weight), weight, rtol=0.03, atol=0.03
        )
        first = linear.weight
        self.assertEqual(replace_linear_weight_rowwise_fp8(linear), 0)
        self.assertIs(linear.weight, first)

        # A later reload must requantize values and scales together.
        reloaded = (weight * 3).contiguous()
        linear.weight.weight_loader(linear.weight, reloaded)
        torch.testing.assert_close(
            dequantize_rowwise_weight(linear.weight), reloaded, rtol=0.03, atol=0.03
        )

    def test_replacement_rejects_unexpected_dtype(self):
        linear = nn.Linear(4, 2, bias=False, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "expected a BF16"):
            replace_linear_weight_rowwise_fp8(linear)

    def test_meta_ingest_installs_resident_fp8_parameter_and_scale(self):
        linear = nn.Linear(4, 2, bias=False, device="meta", dtype=torch.bfloat16)
        self.assertEqual(
            attach_rowwise_ingest([linear], target_device=torch.device("cpu")), 1
        )

        loaded = torch.tensor(
            [[-3.0, -1.0, 1.0, 3.0], [2.0, 2.0, 2.0, 2.0]],
            dtype=torch.bfloat16,
        )
        linear.weight.weight_loader(linear.weight, loaded)
        self.assertEqual(linear.weight.device.type, "cpu")
        self.assertEqual(linear.weight.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(
            dequantize_rowwise_weight(linear.weight), loaded, rtol=0.03, atol=0.03
        )

        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            linear.weight.weight_loader(linear.weight, loaded[:1])
        with self.assertRaisesRegex(RuntimeError, "requires BF16"):
            linear.weight.weight_loader(linear.weight, loaded.float())

    def test_meta_ingest_is_all_or_nothing(self):
        meta = nn.Linear(4, 2, bias=False, device="meta", dtype=torch.bfloat16)
        resident = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "meta-born BF16"):
            attach_rowwise_ingest([meta, resident])
        self.assertFalse(hasattr(meta.weight, "weight_loader"))

    def test_hot_token_selection_preserves_matching_rowwise_scales(self):
        linear = nn.Linear(4, 5, bias=False, dtype=torch.bfloat16)
        linear.weight.data.copy_(torch.arange(20).reshape(5, 4))
        replace_linear_weight_rowwise_fp8(linear)

        token_ids = torch.tensor([4, 1, 3])
        selected = select_rowwise_weight_rows(linear.weight, token_ids)
        self.assertIsInstance(selected, nn.Parameter)
        self.assertFalse(hasattr(selected, "weight_loader"))
        torch.testing.assert_close(
            rowwise_scale_of(selected), rowwise_scale_of(linear.weight)[token_ids]
        )
        torch.testing.assert_close(
            dequantize_rowwise_weight(selected),
            dequantize_rowwise_weight(linear.weight)[token_ids],
        )

    def test_selection_requires_rowwise_metadata(self):
        weight = torch.ones(4, 4, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "rowwise scale metadata"):
            select_rowwise_weight_rows(weight, torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
