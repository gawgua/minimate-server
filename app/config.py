import logging
import math
import os
from dataclasses import dataclass
from decimal import Decimal

from dotenv import load_dotenv

MODEL_ID = "duclvQ/qwen3-asr-1.7b-vimd-full"
INTENT_MODEL_ID = "nguen2929/minimate_intent_classifier"
INTENT_TFIDF_DIR = "models/minimate_synthetic_v1"
INTENT_MODELS = ("bamibert", "tfidf_lr")
RATE = 16000


@dataclass(frozen=True)
class Settings:
    model_id: str
    device: str
    dtype: str
    tokens: int
    min_samples: int
    max_samples: int
    host: str
    port: int
    log_level: str
    intent_enabled: bool
    intent_model_id: str
    highpass_hz: int
    intent_default_model: str
    intent_tfidf_dir: str

    @classmethod
    def load(cls):
        load_dotenv(".env")
        env = os.environ.get
        if int(env("SAMPLE_RATE", "16000")) != RATE:
            raise ValueError("SAMPLE_RATE must be 16000")
        if env("ASR_LANGUAGE", "Vietnamese") != "Vietnamese":
            raise ValueError("ASR_LANGUAGE must be Vietnamese")
        if int(env("MAX_CONCURRENT_INFERENCES", "1")) != 1:
            raise ValueError("MAX_CONCURRENT_INFERENCES must be 1")
        low = Decimal(env("MIN_AUDIO_SECONDS", "0.4")) * RATE
        high = Decimal(env("MAX_AUDIO_SECONDS", "20")) * RATE
        if not low.is_finite() or not high.is_finite():
            raise ValueError("Audio durations must be finite")
        minimum, maximum = math.ceil(low), math.floor(high)
        if not 0 < minimum <= maximum:
            raise ValueError("Invalid audio duration limits")
        tokens = int(env("MAX_NEW_TOKENS", "256"))
        port = int(env("PORT", "8000"))
        level = env("LOG_LEVEL", "INFO").upper()
        if tokens < 1 or not 1 <= port <= 65535 or level not in logging.getLevelNamesMapping():
            raise ValueError("Invalid token limit, port, or log level")
        intent_enabled = env("INTENT_ENABLED", "true").strip().lower()
        if intent_enabled not in {"true", "false"}:
            raise ValueError("INTENT_ENABLED must be true or false")
        highpass_hz = int(env("HIGHPASS_HZ", "100"))
        if not 0 <= highpass_hz <= 300:
            raise ValueError("HIGHPASS_HZ must be between 0 and 300")
        intent_model_id = os.path.expanduser(env("INTENT_MODEL_ID", INTENT_MODEL_ID).strip())
        if intent_enabled == "true" and not intent_model_id:
            raise ValueError("INTENT_MODEL_ID must be set when intent classification is enabled")
        intent_default_model = env("INTENT_DEFAULT_MODEL", "bamibert").strip()
        if intent_default_model not in INTENT_MODELS:
            raise ValueError(f"INTENT_DEFAULT_MODEL must be one of {', '.join(INTENT_MODELS)}")
        intent_tfidf_dir = os.path.expanduser(env("INTENT_TFIDF_DIR", INTENT_TFIDF_DIR).strip())
        return cls(
            env("MODEL_ID", MODEL_ID),
            env("MODEL_DEVICE", "auto"),
            env("MODEL_DTYPE", "auto"),
            tokens,
            minimum,
            maximum,
            env("HOST", "0.0.0.0"),
            port,
            level,
            intent_enabled == "true",
            intent_model_id,
            highpass_hz,
            intent_default_model,
            intent_tfidf_dir,
        )
