import io
import wave

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import RATE


def declick_pcm16(pcm: bytes) -> bytes:
    """Replace short, sharp outliers while keeping the PCM format and length."""
    samples = np.frombuffer(pcm, dtype="<i2")
    if samples.size < 9:
        return pcm
    # An impulse shorter than half this window will not dominate the local median.
    windows = sliding_window_view(samples.astype(np.float32), 9)
    median = np.median(windows, axis=1)
    deviation = np.abs(windows - median[:, None])
    center_deviation = deviation[:, 4].copy()
    local_variation = np.median(deviation, axis=1, overwrite_input=True)
    clicks = center_deviation > np.maximum(4096, 8 * local_variation)
    if not clicks.any():
        return pcm
    cleaned = samples.copy()
    interior = cleaned[4:-4]
    interior[clicks] = median[clicks]
    return cleaned.tobytes()


def highpass_pcm16(pcm: bytes, cutoff_hz: int) -> bytes:
    """Remove DC offset and low rumble, such as supply noise, while keeping the PCM length.

    Zero-phase FFT filtering of the whole utterance. Everything below cutoff_hz is removed,
    and the gain rises along a raised cosine to full level at 1.25 * cutoff_hz.
    cutoff_hz=0 returns the input unchanged.
    """
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    if not cutoff_hz or samples.size < 2:
        return pcm
    # Reflect-pad so the circular FFT does not wrap the end of the clip onto its start.
    pad = min(samples.size - 1, RATE // 10)
    padded = np.pad(samples, pad, mode="reflect")
    freqs = np.fft.rfftfreq(padded.size, 1 / RATE)
    ramp = np.clip((freqs - cutoff_hz) / (0.25 * cutoff_hz), 0, 1)
    gain = 0.5 - 0.5 * np.cos(np.pi * ramp)
    filtered = np.fft.irfft(np.fft.rfft(padded) * gain, n=padded.size)[pad : pad + samples.size]
    return np.clip(np.rint(filtered), -32768, 32767).astype("<i2").tobytes()


def clean_pcm16(pcm: bytes, highpass_hz: int) -> bytes:
    return highpass_pcm16(declick_pcm16(pcm), highpass_hz)


def pcm16_to_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE)
        audio.writeframes(pcm)
    return output.getvalue()


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:
        raise ValueError("PCM16 requires even byte count")
    if not pcm:
        raise ValueError("Audio too short")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    samples /= 32768.0
    return samples
