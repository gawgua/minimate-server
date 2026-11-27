import pytest

from app.config import INTENT_MODEL_ID, Settings
from app.protocol import Recording


def test_limits():
    rec = Recording(2, 4)
    for n in (4, 8):
        rec.append(bytes(n))
        assert rec.finish() == bytes(n)
    rec.append(bytes(2))
    with pytest.raises(ValueError, match="too short"):
        rec.finish()
    with pytest.raises(ValueError, match="too long"):
        rec.append(bytes(10))
    rec.append(bytes(6))
    assert rec.finish() is None


def test_defaults():
    settings = Settings.load()
    assert (settings.min_samples, settings.max_samples, settings.tokens) == (6400, 320000, 256)
    assert settings.device == settings.dtype == "auto"
    assert settings.intent_enabled and settings.intent_model_id == INTENT_MODEL_ID
    assert settings.highpass_hz == 100


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("SAMPLE_RATE", "8000"),
        ("MAX_CONCURRENT_INFERENCES", "2"),
        ("ASR_LANGUAGE", "English"),
        ("MIN_AUDIO_SECONDS", "nan"),
        ("MAX_AUDIO_SECONDS", "0.1"),
        ("INTENT_ENABLED", "maybe"),
        ("INTENT_MODEL_ID", ""),
        ("HIGHPASS_HZ", "-1"),
        ("HIGHPASS_HZ", "301"),
    ],
)
def test_invalid_env(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        Settings.load()
