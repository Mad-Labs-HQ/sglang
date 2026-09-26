"""Actual-kernel numerics and CUDA-graph coverage for SM120 online FP8.

Covers the rowwise-FP8 lm_head GEMV and the FlashInfer CUTLASS MXFP8 dense
path that SGLANG_SM120_ONLINE_MXFP8 installs on Flash-Next's BF16 projections.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    dequantize_rowwise_weight,
    replace_linear_weight_rowwise_fp8,
    rowwise_fp8_lm_head_logits,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


def _is_exact_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(
    not _is_exact_sm120(), reason="online FP8 kernels require exactly SM120"
)

HIDDEN_SIZE = 2560
VOCAB_ROWS = 1024

# Flash-Next (Qwen3.8-Flash-Next) projections the switch converts at TP1.
FLASH_NEXT_MXFP8_SHAPES = [
    pytest.param(16384, 2560, id="gdn-in_proj_qkvz"),
    pytest.param(2560, 6144, id="gdn-out_proj-and-o_proj"),
    pytest.param(13312, 2560, id="attn-qkv_proj"),
    pytest.param(640, 2560, id="qsa-index_qk_proj"),
    pytest.param(1280, 2560, id="shared_expert-gate_up_proj"),
    pytest.param(2560, 640, id="shared_expert-down_proj"),
]


def _randn(shape, *, seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(
            shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * scale
    )


def _assert_normalized_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_nrmse: float,
    min_cosine: float,
) -> None:
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
def rowwise_lm_head_weight() -> torch.nn.Parameter:
    linear = nn.Linear(
        HIDDEN_SIZE,
        VOCAB_ROWS,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    linear.weight.data.copy_(
        _randn(
            (VOCAB_ROWS, HIDDEN_SIZE),
            seed=101,
            scale=1.0 / math.sqrt(HIDDEN_SIZE),
        )
    )
    replace_linear_weight_rowwise_fp8(linear)
    return linear.weight


def _rowwise_reference(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dense_weight = dequantize_rowwise_weight(weight, torch.bfloat16)
    return hidden.bfloat16() @ dense_weight.T


@pytest.mark.parametrize("rows", [1, 4, 16, 24, 33])
def test_rowwise_lm_head_matches_dequantized_bf16_reference(
    rowwise_lm_head_weight: torch.Tensor, rows: int
) -> None:
    hidden = _randn((rows, HIDDEN_SIZE), seed=200 + rows, scale=0.25)

    actual = rowwise_fp8_lm_head_logits(hidden, rowwise_lm_head_weight)
    expected = _rowwise_reference(hidden, rowwise_lm_head_weight)

    _assert_normalized_error(actual, expected, max_nrmse=0.025, min_cosine=0.999)


@pytest.mark.parametrize("rows", [1, 4, 16, 24, 33])
def test_rowwise_lm_head_cuda_graph_replays_mutated_input(
    rowwise_lm_head_weight: torch.Tensor, rows: int
) -> None:
    static_hidden = _randn((rows, HIDDEN_SIZE), seed=300 + rows, scale=0.25)

    # Compile/JIT and populate allocator state before capture.
    for _ in range(2):
        rowwise_fp8_lm_head_logits(static_hidden, rowwise_lm_head_weight)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = rowwise_fp8_lm_head_logits(static_hidden, rowwise_lm_head_weight)

    changed_hidden = _randn((rows, HIDDEN_SIZE), seed=400 + rows, scale=0.25)
    before = _rowwise_reference(static_hidden, rowwise_lm_head_weight)
    static_hidden.copy_(changed_hidden)
    expected = _rowwise_reference(changed_hidden, rowwise_lm_head_weight)
    assert not torch.equal(before, expected)

    graph.replay()
    torch.cuda.synchronize()
    actual = graph_output.clone()

    _assert_normalized_error(actual, expected, max_nrmse=0.025, min_cosine=0.999)


class _Dense(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)


@pytest.fixture(
    params=["cutlass", "flashinfer_cutlass"],
    ids=["sm120-auto-resolves-cutlass", "explicit-flashinfer-cutlass"],
)
def online_mxfp8_method(request):
    """The online MXFP8 method, as the Flash-Next switch builds it."""
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        Fp8GemmRunnerBackend,
        Mxfp8DenseGemmBackend,
    )

    original_backend = fp8_utils.FP8_GEMM_RUNNER_BACKEND
    # initialize_fp8_gemm_config maps --fp8-gemm-backend=auto to "cutlass" on
    # SM120; the MXFP8 resolver must still land on FlashInfer CUTLASS.
    fp8_utils.FP8_GEMM_RUNNER_BACKEND = Fp8GemmRunnerBackend(request.param)
    try:
        method = Fp8LinearMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=False,
                activation_scheme="dynamic",
                use_mxfp8=True,
            )
        )
        assert method.mxfp8_dense_backend is Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS
        yield method
    finally:
        fp8_utils.FP8_GEMM_RUNNER_BACKEND = original_backend


@pytest.mark.parametrize("n, k", FLASH_NEXT_MXFP8_SHAPES)
def test_flashinfer_cutlass_mxfp8_linear_quantizes_and_applies(
    online_mxfp8_method, n: int, k: int
) -> None:
    original_weight = _randn((n, k), seed=501 + n + k, scale=1.0 / math.sqrt(k))
    layer = _Dense(original_weight.clone())
    online_mxfp8_method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale_inv.dtype == torch.uint8
    assert layer.weight_scale_inv.format_ue8m0

    for rows in (1, 4, 16, 24, 257):
        x = _randn((rows, k), seed=500 + rows, scale=0.25)
        actual = online_mxfp8_method.apply(layer, x)
        expected = x.float() @ original_weight.float().T
        _assert_normalized_error(actual, expected, max_nrmse=0.12, min_cosine=0.99)


def test_mxfp8_linear_cuda_graph_replays_mutated_input(online_mxfp8_method) -> None:
    n, k, rows = 2560, 6144, 4
    original_weight = _randn((n, k), seed=601, scale=1.0 / math.sqrt(k))
    layer = _Dense(original_weight.clone())
    online_mxfp8_method.process_weights_after_loading(layer)

    static_x = _randn((rows, k), seed=602, scale=0.25)
    for _ in range(2):
        online_mxfp8_method.apply(layer, static_x)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = online_mxfp8_method.apply(layer, static_x)

    changed_x = _randn((rows, k), seed=603, scale=0.25)
    static_x.copy_(changed_x)
    graph.replay()
    torch.cuda.synchronize()

    expected = changed_x.float() @ original_weight.float().T
    _assert_normalized_error(
        graph_output.clone(), expected, max_nrmse=0.12, min_cosine=0.99
    )
