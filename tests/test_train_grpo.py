import pytest

from train_grpo import resolve_precision


@pytest.mark.parametrize(
    ("cuda_available", "bf16_supported", "expected"),
    [
        (False, False, "fp32"),
        (True, False, "fp16"),
        (True, True, "bf16"),
    ],
)
def test_auto_precision(cuda_available, bf16_supported, expected):
    assert resolve_precision("auto", cuda_available, bf16_supported) == expected


def test_bf16_requires_hardware_support():
    with pytest.raises(ValueError, match="does not support"):
        resolve_precision("bf16", cuda_available=True, bf16_supported=False)


def test_half_precision_requires_cuda():
    with pytest.raises(ValueError, match="CUDA accelerator"):
        resolve_precision("fp16", cuda_available=False, bf16_supported=False)
