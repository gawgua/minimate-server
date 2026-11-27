import hashlib
import importlib
import io
import json
import logging
import math
import re
import unicodedata
import warnings
from pathlib import Path

log = logging.getLogger(__name__)


def no_progress(fraction: float, stage: str):
    pass


class IntentClassifier:
    """Fine-tuned BamiBERT on CPU.

    An optional clarify_threshold in config.json, picked on validation during training, makes
    predictions below it send no intent. Without it, every prediction is used.
    """

    def __init__(self, model, tokenizer):
        config = model.config
        threshold = getattr(config, "clarify_threshold", None)
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("clarify_threshold in the intent config must be between 0 and 1")
        self.threshold = threshold
        labels = config.id2label
        if config.num_labels < 2 or set(labels or {}) != set(range(config.num_labels)):
            raise ValueError("Intent model must provide an id2label entry for each class")
        if (
            any(
                not isinstance(label, str) or not label.strip() or label == f"LABEL_{index}"
                for index, label in labels.items()
            )
            or len(set(labels.values())) != config.num_labels
        ):
            raise ValueError("Intent model must provide distinct intent names in id2label")
        if config.problem_type not in {None, "single_label_classification"}:
            raise ValueError("Intent model must be a single-label classifier")

        # RoBERTa starts non-padding position IDs immediately after pad_token_id.
        model_limit = config.max_position_embeddings - config.pad_token_id - 1
        self.max_length = min(tokenizer.model_max_length, model_limit)
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels

    @classmethod
    def load(cls, model_id: str, progress=no_progress):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from .checkpoint import intent_checkpoint

        progress(0.0, "Checking for updates")
        # Resolve one commit so the tokenizer and weights always come from the same revision.
        model_id = intent_checkpoint(
            model_id, progress=lambda done: progress(0.6 * done, "Downloading files")
        )
        progress(0.6, "Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        progress(0.7, "Loading weights")
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=False, output_loading_info=True
        )
        if loading_info["missing_keys"] or loading_info["mismatched_keys"]:
            raise ValueError(
                "Intent checkpoint is missing trained weights or has incompatible shapes"
            )
        return cls(model.to("cpu").eval(), tokenizer)

    def classify(self, text: str) -> dict[str, int | str | float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        logits = self.model(**inputs).logits
        score, index = logits.float().softmax(dim=-1).max(dim=-1)
        confidence = score.item()
        if not math.isfinite(confidence):
            raise ValueError("Intent classifier produced a non-finite score")
        label_id = index.item()
        return {"id": label_id, "label": self.labels[label_id], "score": confidence}


def tfidf_normalize(text: str) -> str:
    """Same as normalize() in the TF-IDF training code: NFC, lowercase, punctuation to spaces."""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# Everything the TF-IDF pipeline pickle may import. Unpickling runs code, so anything else is
# refused. The training code's common.normalize resolves to the copy above.
TFIDF_GLOBALS = {
    ("common", "normalize"): tfidf_normalize,
    **{
        (module, name): None
        for module, name in (
            ("joblib.numpy_pickle", "NumpyArrayWrapper"),
            ("numpy", "dtype"),
            ("numpy", "float64"),
            ("numpy", "ndarray"),
            ("sklearn.feature_extraction.text", "TfidfTransformer"),
            ("sklearn.feature_extraction.text", "TfidfVectorizer"),
            ("sklearn.linear_model._logistic", "LogisticRegression"),
            ("sklearn.pipeline", "Pipeline"),
        )
    },
}


def load_tfidf_pipeline(data: bytes, filename: str):
    from joblib.numpy_pickle import NumpyUnpickler
    from joblib.numpy_pickle_utils import _validate_fileobject_and_memmap

    class AllowlistUnpickler(NumpyUnpickler):
        def find_class(self, module, name):
            if (module, name) not in TFIDF_GLOBALS:
                raise ValueError(f"Intent pipeline references disallowed {module}.{name}")
            return TFIDF_GLOBALS[module, name] or getattr(importlib.import_module(module), name)

    # Unpickle the exact bytes whose checksum was verified.
    with _validate_fileobject_and_memmap(io.BytesIO(data), filename, None) as (fobj, _):
        return AllowlistUnpickler(filename, fobj, True).load()


class TfidfIntentClassifier:
    """TF-IDF + logistic regression export with its validated confidence threshold."""

    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold
        self.labels = [str(label) for label in model.classes_]

    @classmethod
    def load(cls, artifacts_dir: str, progress=no_progress):
        directory = Path(artifacts_dir).expanduser()
        progress(0.0, "Reading config")
        config = json.loads((directory / "inference_config.json").read_text(encoding="utf-8"))
        if not config.get("constraints_met"):
            raise ValueError("Intent pipeline threshold has not passed validation")
        progress(0.2, "Verifying checksum")
        path = directory / config["model_file"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != config["model_sha256"]:
            raise ValueError("Intent pipeline does not match the checksum in inference_config.json")
        progress(0.5, "Loading weights")
        from sklearn.exceptions import InconsistentVersionWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = load_tfidf_pipeline(data, str(path))
        # sklearn warns once per estimator on a version mismatch. Log it once instead.
        saved = set()
        for warning in caught:
            if isinstance(warning.message, InconsistentVersionWarning):
                saved.add(warning.message.original_sklearn_version)
            else:
                warnings.warn_explicit(
                    warning.message, warning.category, warning.filename, warning.lineno
                )
        for version in saved:
            log.warning(
                "TF-IDF pipeline was saved with scikit-learn %s; this server runs a "
                "different version",
                version,
            )
        if {str(label) for label in model.classes_} != set(config["labels"]):
            raise ValueError("Intent pipeline labels do not match inference_config.json")
        threshold = float(config["threshold"])
        if not 0 <= threshold <= 1:
            raise ValueError("Intent pipeline threshold must be between 0 and 1")
        return cls(model, threshold)

    def classify(self, text: str) -> dict[str, str | float]:
        probabilities = self.model.predict_proba([text])[0]
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        if not math.isfinite(confidence):
            raise ValueError("Intent classifier produced a non-finite score")
        return {"label": self.labels[index], "score": confidence}
