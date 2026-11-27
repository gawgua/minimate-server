import gc
import logging
import threading
from pathlib import Path

from .intent import IntentClassifier, TfidfIntentClassifier

log = logging.getLogger(__name__)

MODEL_NAMES = {"bamibert": "BamiBERT", "tfidf_lr": "TF-IDF + Logistic Regression"}
WARM_UP_TEXT = "Xin chào."


class IntentSwitcher:
    """Holds at most one intent model in memory and swaps it in the background.

    Switching unloads the active model first, so two models never share memory.
    While a load runs, transcripts get no intent and further switches are refused.
    """

    def __init__(self, settings=None, classifier=None, active=None):
        self.settings = settings
        self._lock = threading.Lock()
        self.classifier = classifier
        self.active = active if classifier is not None else None
        self.loading = None
        self.progress = 0
        self.stage = ""
        self.error = None
        self.closed = False

    @property
    def enabled(self) -> bool:
        return self.settings is not None and self.settings.intent_enabled

    def available(self, key: str) -> bool:
        if key == "tfidf_lr":
            return (Path(self.settings.intent_tfidf_dir) / "inference_config.json").is_file()
        return key in MODEL_NAMES

    def current(self):
        with self._lock:
            return self.classifier, self.active

    def set(self, classifier, active="bamibert"):
        with self._lock:
            self.classifier = classifier
            self.active = active if classifier is not None else None

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "active": self.active,
                "loading": self.loading,
                "progress": self.progress,
                "stage": self.stage,
                "error": self.error,
                "options": [
                    {
                        "key": key,
                        "name": name,
                        "available": self.settings is not None and self.available(key),
                    }
                    for key, name in MODEL_NAMES.items()
                ],
            }

    def _report(self, fraction: float, stage: str):
        # Loading covers 0-95%; the warm-up classification finishes the bar.
        with self._lock:
            self.progress = max(self.progress, min(95, int(95 * fraction)))
            self.stage = stage

    def _begin(self, key: str):
        """Claim the loader and unload the active model. Caller holds no lock."""
        with self._lock:
            if self.loading is not None:
                return False
            self.loading = key
            self.progress = 0
            self.stage = "Unloading previous model"
            self.error = None
            self.classifier = None
            self.active = None
        gc.collect()  # Free the previous model before the next one allocates.
        return True

    def _load(self, key: str):
        if key == "tfidf_lr":
            classifier = TfidfIntentClassifier.load(self.settings.intent_tfidf_dir, self._report)
        else:
            classifier = IntentClassifier.load(self.settings.intent_model_id, self._report)
        self._report(0.95, "Warming up")
        classifier.classify(WARM_UP_TEXT)
        return classifier

    def _finish(self, key: str, classifier=None, error=None):
        with self._lock:
            self.loading = None
            if self.closed:
                classifier = None
            elif error is None:
                self.classifier, self.active = classifier, key
                self.progress, self.stage = 100, "Ready"
            else:
                self.error = f"Could not load {MODEL_NAMES[key]}: {error}"[:300]
                self.progress, self.stage = 0, ""

    def load_now(self, key: str):
        """Load synchronously, for startup. Errors propagate."""
        if not self._begin(key):
            raise RuntimeError("Another intent model is loading")
        try:
            classifier = self._load(key)
        except Exception as exc:
            self._finish(key, error=exc)
            raise
        self._finish(key, classifier)
        log.info("Intent model ready: %s", MODEL_NAMES[key])

    def start(self, key: str) -> bool:
        """Switch in a background thread. Returns False when a load is already running."""
        if not self.enabled:
            raise ValueError("Intent classification is disabled")
        if key not in MODEL_NAMES or not self.available(key):
            raise ValueError(f"Unknown or unavailable intent model: {key}")
        if not self._begin(key):
            return False
        log.info("Switching intent model to %s", MODEL_NAMES[key])

        def run():
            try:
                classifier = self._load(key)
            except Exception as exc:
                log.exception("Could not load intent model %s", MODEL_NAMES[key])
                self._finish(key, error=exc)
            else:
                self._finish(key, classifier)
                log.info("Intent model ready: %s", MODEL_NAMES[key])

        threading.Thread(target=run, name=f"intent-load-{key}", daemon=True).start()
        return True

    def close(self):
        with self._lock:
            self.closed = True
            self.classifier = None
            self.active = None
