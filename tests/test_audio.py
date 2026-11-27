import struct

import numpy as np
import pytest

from app.audio import clean_pcm16, declick_pcm16, highpass_pcm16, pcm16_to_float32
from app.config import RATE


def test_samples():
    values = [-32768, -1, 0, 256, 32767]
    samples = pcm16_to_float32(struct.pack("<5h", *values))
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(samples, np.array(values) / 32768)


@pytest.mark.parametrize("pcm", [b"", b"x"])
def test_invalid(pcm):
    with pytest.raises(ValueError):
        pcm16_to_float32(pcm)


def test_declick_short_burst_without_changing_smooth_audio():
    smooth = np.rint(5000 * np.sin(np.arange(800) * 0.09)).astype("<i2")
    assert declick_pcm16(smooth.tobytes()) == smooth.tobytes()
    damaged = smooth.copy()
    damaged[395:398] = 32000
    cleaned = np.frombuffer(declick_pcm16(damaged.tobytes()), dtype="<i2")
    np.testing.assert_array_equal(cleaned[:395], smooth[:395])
    np.testing.assert_array_equal(cleaned[398:], smooth[398:])
    assert np.max(np.abs(cleaned[395:398] - smooth[395:398])) < 1500


def tone_level(samples, hz):
    """Amplitude of one frequency, measured away from the clip edges."""
    middle = samples[RATE // 4 : -RATE // 4].astype(np.float64)
    t = np.arange(middle.size) / RATE
    return 2 * np.abs(np.mean(middle * np.exp(-2j * np.pi * hz * t)))


def test_highpass_removes_rumble_and_keeps_speech_band():
    t = np.arange(RATE * 2) / RATE
    speech = 4000 * np.sin(2 * np.pi * 1000 * t) + 3000 * np.sin(2 * np.pi * 300 * t)
    # Supply noise measured on the ESP32: about 20 Hz and 84 Hz, plus a DC offset.
    rumble = 2000 * np.sin(2 * np.pi * 20 * t) + 2000 * np.sin(2 * np.pi * 84 * t)
    noisy = np.rint(speech + rumble - 3000).astype("<i2")
    cleaned = np.frombuffer(highpass_pcm16(noisy.tobytes(), 100), dtype="<i2")
    assert cleaned.size == noisy.size
    assert tone_level(cleaned, 20) < 2000 / 100  # At least 40 dB less.
    assert tone_level(cleaned, 84) < 2000 / 100
    assert tone_level(cleaned, 1000) == pytest.approx(4000, rel=0.01)
    assert tone_level(cleaned, 300) == pytest.approx(3000, rel=0.01)
    assert abs(cleaned[RATE // 4 : -RATE // 4].mean()) < 20  # DC offset removed.


def test_highpass_disabled_and_tiny_inputs_pass_through():
    pcm = np.arange(-50, 50, dtype="<i2").tobytes()
    assert highpass_pcm16(pcm, 0) == pcm
    assert highpass_pcm16(b"\x01\x00", 100) == b"\x01\x00"
    assert clean_pcm16(bytes(1600), 100) == bytes(1600)
