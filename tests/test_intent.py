import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.intent import IntentClassifier


@pytest.fixture
def export(monkeypatch, tmp_path):
    config = SimpleNamespace(
        id2label={0: "greeting", 1: "weather"},
        num_labels=2,
        problem_type="single_label_classification",
        max_position_embeddings=2050,
        pad_token_id=1,
    )
    model = Mock(config=config)
    model.to.return_value.eval.return_value = model
    tokenizer = Mock(model_max_length=4096, return_value={"input_ids": object()})
    info = {"missing_keys": [], "mismatched_keys": []}
    load_model = Mock(return_value=(model, info))
    load_tokenizer = Mock(return_value=tokenizer)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=load_model),
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        ),
    )
    return SimpleNamespace(
        path=str(tmp_path),
        model=model,
        tokenizer=tokenizer,
        load_model=load_model,
        load_tokenizer=load_tokenizer,
        info=info,
    )


def test_load_and_score(export):
    classifier = IntentClassifier.load(export.path)
    export.load_model.assert_called_once_with(
        export.path, trust_remote_code=False, output_loading_info=True
    )
    export.load_tokenizer.assert_called_once_with(export.path, trust_remote_code=False)
    export.model.to.assert_called_once_with("cpu")
    export.model.to.return_value.eval.assert_called_once()

    probabilities = export.model.return_value.logits.float.return_value.softmax
    top = probabilities.return_value.max
    top.return_value = Mock(item=Mock(return_value=0.8)), Mock(item=Mock(return_value=1))
    assert classifier.classify("Trời mưa không?") == {"id": 1, "label": "weather", "score": 0.8}
    probabilities.assert_called_once_with(dim=-1)
    top.assert_called_once_with(dim=-1)
    export.tokenizer.assert_called_once_with(
        "Trời mưa không?", return_tensors="pt", truncation=True, max_length=2048
    )
    export.model.assert_called_once_with(**export.tokenizer.return_value)


@pytest.mark.parametrize("labels", [{0: "greeting"}, {0: "LABEL_0", 1: "LABEL_1"}])
def test_reject_invalid_labels(export, labels):
    export.model.config.id2label = labels
    with pytest.raises(ValueError, match="id2label"):
        IntentClassifier.load(export.path)


def test_reject_missing_classifier_weights(export):
    export.info["missing_keys"] = ["classifier.out_proj.weight"]
    with pytest.raises(ValueError, match="missing trained weights"):
        IntentClassifier.load(export.path)


def test_clarify_threshold_from_config(export):
    assert IntentClassifier.load(export.path).threshold is None
    export.model.config.clarify_threshold = 0.55
    assert IntentClassifier.load(export.path).threshold == 0.55


@pytest.mark.parametrize("threshold", [1.5, -0.1, "0.5", True])
def test_reject_invalid_clarify_threshold(export, threshold):
    export.model.config.clarify_threshold = threshold
    with pytest.raises(ValueError, match="clarify_threshold"):
        IntentClassifier.load(export.path)
