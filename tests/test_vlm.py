import pytest
import torch

from docextract.vlm import _dtype


@pytest.mark.parametrize(("capability", "expected"), [
    ((7, 5), torch.float16),   # T4: bf16 only emulated, even though is_bf16_supported() says True
    ((8, 0), torch.bfloat16),  # A100
    ((8, 9), torch.bfloat16),  # L4
])
def test_cuda_dtype_follows_compute_capability(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    assert _dtype("cuda") == expected


def test_non_cuda_devices_use_bf16():
    assert _dtype("mps") == torch.bfloat16
    assert _dtype("cpu") == torch.bfloat16
