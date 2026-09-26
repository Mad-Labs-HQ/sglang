"""SM120 online FP8: rowwise-FP8 HyperConnection mix weights, on real kernels.

Decode rows (<= 16) run the persistent fused Triton kernel with FP8 weights;
prefill rows fall back to the torch.compile path on transient BF16 operands.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    configure_online_fp8,
    dequantize_rowwise_weight,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


def _is_exact_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(
    not _is_exact_sm120(), reason="online FP8 kernels require exactly SM120"
)

HIDDEN_SIZE = 2560
HC_COUNT = 4
HC_LOWRANK = 320


def _randn(shape, *, seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        * scale
    )


def _assert_normalized_error(actual, expected, *, max_nrmse, min_cosine) -> None:
    assert actual.shape == expected.shape
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    error_rms = (actual_fp32 - expected_fp32).square().mean().sqrt().item()
    reference_rms = expected_fp32.square().mean().sqrt().item()
    nrmse = error_rms / max(reference_rms, 1e-8)
    cosine = F.cosine_similarity(
        actual_fp32.flatten(), expected_fp32.flatten(), dim=0
    ).item()
    assert nrmse <= max_nrmse, f"NRMSE {nrmse:.6f} exceeds {max_nrmse:.6f}"
    assert cosine >= min_cosine, f"cosine {cosine:.6f} is below {min_cosine:.6f}"


@pytest.fixture(scope="module")
def rowwise_hyperconnection():
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    configure_online_fp8(
        True,
        cuda_available=True,
        capability=torch.cuda.get_device_capability(),
    )
    try:
        config = HyperConnectionConfig(
            hc_count=HC_COUNT,
            hidden_size=HIDDEN_SIZE,
            params_dtype=torch.bfloat16,
            hc_lowrank=HC_LOWRANK,
            hc_per_branch_norm=True,
        )
        with torch.device("cuda"):
            layer = GatedResidual(
                config, use_mix=True, use_combine=False, online_fp8=True
            )
        # Model construction runs under the configured BF16 default dtype.
        # Reproduce that for the norm parameter without touching FP8 mix weights.
        layer.hc_norm.to(dtype=torch.bfloat16)
        assert layer._online_fp8_mix
        assert layer.input_mix_weight_down.weight.is_meta

        down = _randn(
            (HC_LOWRANK, HC_COUNT * HIDDEN_SIZE),
            seed=601,
            scale=1.0 / math.sqrt(HC_COUNT * HIDDEN_SIZE),
        )
        up = _randn(
            (HC_COUNT * HIDDEN_SIZE, HC_LOWRANK),
            seed=602,
            scale=1.0 / math.sqrt(HC_LOWRANK),
        )
        # Checkpoint tensors arrive on the host.
        down_loader = layer.input_mix_weight_down.weight.weight_loader
        up_loader = layer.input_mix_weight_up.weight.weight_loader
        down_loader(layer.input_mix_weight_down.weight, down.cpu())
        up_loader(layer.input_mix_weight_up.weight, up.cpu())
        assert layer.input_mix_weight_down.weight.dtype == torch.float8_e4m3fn
        assert layer.input_mix_weight_up.weight.dtype == torch.float8_e4m3fn
        assert layer.input_mix_weight_down.weight.is_cuda
        del down, up, down_loader, up_loader
        yield layer
    finally:
        configure_online_fp8(False, cuda_available=True, capability=(12, 0))


def _hyperconnection_reference(layer, hyper_input: torch.Tensor) -> torch.Tensor:
    normed = layer.hc_norm(hyper_input)
    down = dequantize_rowwise_weight(layer.input_mix_weight_down.weight, torch.bfloat16)
    up = dequantize_rowwise_weight(layer.input_mix_weight_up.weight, torch.bfloat16)
    mix = F.silu(F.linear(normed, down) / HC_COUNT)
    mix = torch.sigmoid(F.linear(mix, up)).unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
    return (mix * normed.unflatten(-1, (HC_COUNT, HIDDEN_SIZE))).mean(dim=-2)


@pytest.mark.parametrize("rows", [1, 4, 16, 33])
def test_rowwise_hyperconnection_fused_and_prefill_paths(
    rowwise_hyperconnection, rows: int
) -> None:
    from sglang.srt.layers.hc_mix_triton import fused_hc_mix_supported

    hyper_input = _randn((rows, HC_COUNT * HIDDEN_SIZE), seed=700 + rows, scale=0.25)
    normed = rowwise_hyperconnection.hc_norm(hyper_input)
    uses_fused_kernel = fused_hc_mix_supported(
        normed,
        rowwise_hyperconnection.input_mix_weight_down.weight,
        rowwise_hyperconnection.input_mix_weight_up.weight,
    )
    assert uses_fused_kernel is (rows <= 16)

    actual, _ = rowwise_hyperconnection.mix(hyper_input)
    expected = _hyperconnection_reference(rowwise_hyperconnection, hyper_input)

    # Decode rows exercise the persistent fused kernel. At 33 rows the actual
    # GatedResidual path dequantizes transient BF16 operands for prefill.
    _assert_normalized_error(actual, expected, max_nrmse=0.06, min_cosine=0.995)


def test_rowwise_hyperconnection_cuda_graph_replays_mutated_input(
    rowwise_hyperconnection,
) -> None:
    rows = 4
    static_input = _randn((rows, HC_COUNT * HIDDEN_SIZE), seed=801, scale=0.25)
    for _ in range(2):
        rowwise_hyperconnection.mix(static_input)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, _ = rowwise_hyperconnection.mix(static_input)

    changed = _randn((rows, HC_COUNT * HIDDEN_SIZE), seed=802, scale=0.25)
    static_input.copy_(changed)
    graph.replay()
    torch.cuda.synchronize()

    expected = _hyperconnection_reference(rowwise_hyperconnection, changed)
    _assert_normalized_error(
        graph_output.clone(), expected, max_nrmse=0.06, min_cosine=0.995
    )
