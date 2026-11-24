import sys
from types import SimpleNamespace

import pytest

from docextract.vlm import _dtype


def fake_torch(capability):
    """Stand-in for torch (not installed in CI): just what _dtype touches."""
    cuda = SimpleNamespace(get_device_capability=lambda *a: capability,
                           # True on a T4 too, because bf16 can be emulated there
                           is_bf16_supported=lambda *a, **k: True)
    return SimpleNamespace(cuda=cuda, float16="float16", bfloat16="bfloat16")


@pytest.mark.parametrize(("capability", "expected"), [
    ((7, 5), "float16"),   # T4: no bf16 tensor cores
    ((8, 0), "bfloat16"),  # A100
    ((8, 9), "bfloat16"),  # L4
])
def test_cuda_dtype_follows_compute_capability(monkeypatch, capability, expected):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(capability))
    assert _dtype("cuda") == expected


def test_non_cuda_devices_use_bf16(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch((7, 5)))
    assert _dtype("mps") == "bfloat16"
    assert _dtype("cpu") == "bfloat16"
