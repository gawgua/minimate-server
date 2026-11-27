import asyncio
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.asr import clean_transcript, select_device


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" language Vietnamese<asr_text>Xin chào! ", "Xin chào!"),
        ("language English<asr_text>text", "language English<asr_text>text"),
        ("ừ ừ, 22.", "ừ ừ, 22."),
        ("", ""),
        ("x language Vietnamese<asr_text>", "x language Vietnamese<asr_text>"),
    ],
)
def test_clean(value, expected):
    assert clean_transcript(value) == expected


@pytest.mark.parametrize(
    ("available", "bf16", "want"),
    [(True, True, "bf16"), (True, False, "fp16"), (False, False, "fp32")],
)
def test_device(available, bf16, want):
    cuda = Mock()
    cuda.is_available.return_value = available
    cuda.device_count.return_value = 1
    cuda.device.return_value = nullcontext()
    cuda.is_bf16_supported.return_value = bf16
    torch = SimpleNamespace(cuda=cuda, bfloat16="bf16", float16="fp16", float32="fp32")
    device, dtype = select_device(torch, "auto", "auto")
    assert dtype == want and device == ("cuda:0" if available else "cpu")
    assert select_device(torch, "cpu", "bf16") == ("cpu", "bf16")
    with pytest.raises(ValueError):
        select_device(torch, "cuda:7", "auto")
    with pytest.raises(ValueError):
        select_device(torch, "cpu", "fp16")


def test_transcribe(service):
    samples = np.zeros(6400, dtype=np.float32)
    assert asyncio.run(service.transcribe(samples))["text"] == "Xin chào"
    service.model.transcribe.assert_called_once_with(audio=(samples, 16000), language="Vietnamese")
    service.model.transcribe.return_value = [SimpleNamespace(text="")]
    assert asyncio.run(service.transcribe(samples))["text"] == ""
    service.model.transcribe.side_effect = RuntimeError("inference")
    with pytest.raises(RuntimeError, match="inference"):
        asyncio.run(service.transcribe(samples))


def test_cancelled_waiter_and_shutdown(service):
    started, release = threading.Event(), threading.Event()

    def slow(**kwargs):
        started.set()
        assert release.wait(3)
        return [SimpleNamespace(text="")]

    service.model.transcribe.side_effect = slow

    async def scenario():
        first = asyncio.create_task(service.transcribe([]))
        assert await asyncio.to_thread(started.wait, 3)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(service.transcribe([]))
        await asyncio.sleep(0)
        shutdown = asyncio.create_task(service.close())
        await asyncio.sleep(0)
        assert service.model.transcribe.call_count == 1 and not shutdown.done()
        release.set()
        with pytest.raises(RuntimeError, match="unavailable"):
            await second
        await shutdown
        assert service.model is None

    asyncio.run(scenario())


def test_gpu_metrics_and_oom(service):
    service.device = "cuda:1"
    cuda = Mock()
    service.torch.cuda = cuda
    service.torch.OutOfMemoryError = MemoryError
    cuda.memory_allocated.return_value = 10
    cuda.max_memory_allocated.return_value = 20
    result = asyncio.run(service.transcribe([]))
    assert result["gpu_allocated_bytes"] == 10 and result["gpu_peak_bytes"] == 20
    cuda.reset_peak_memory_stats.assert_called_once_with("cuda:1")
    assert cuda.synchronize.call_count == 2
    service.model.transcribe.side_effect = MemoryError("OOM")
    with pytest.raises(MemoryError):
        asyncio.run(service.transcribe([]))
    assert service.device == "cuda:1"
