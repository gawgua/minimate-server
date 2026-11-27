import collections
import hashlib
import json
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.intent import IntentClassifier, TfidfIntentClassifier, tfidf_normalize
from app.intent_models import IntentSwitcher

PCM = bytes(12800)
LABELS = ["greeting", "out_of_scope", "schedule_reminder"]


def write_export(directory, model, **config):
    path = directory / "intent_pipeline.joblib"
    joblib.dump(model, path)
    config = {
        "model_file": path.name,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "labels": LABELS,
        "threshold": 0.4,
        "constraints_met": True,
        **config,
    }
    (directory / "inference_config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


@pytest.fixture
def export(tmp_path, monkeypatch):
    # The real export pickles common.normalize from the training code. Recreate that reference.
    common = ModuleType("common")

    def normalize(text):
        return tfidf_normalize(text)

    normalize.__module__, normalize.__qualname__ = "common", "normalize"
    common.normalize = normalize
    monkeypatch.setitem(sys.modules, "common", common)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=normalize, lowercase=False)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    model.fit(
        ["xin chào", "chào bạn", "mở nhạc", "bật đèn", "nhắc tôi lúc 3 giờ", "nhắc mình mai"],
        ["greeting", "greeting", "out_of_scope", "out_of_scope"]
        + ["schedule_reminder", "schedule_reminder"],
    )
    write_export(tmp_path, model)
    monkeypatch.delitem(sys.modules, "common")  # Loading must not need the training code.
    return tmp_path


def test_tfidf_export_loads_through_allowlist(export):
    progress = []
    classifier = TfidfIntentClassifier.load(str(export), lambda f, stage: progress.append(f))
    assert classifier.threshold == 0.4
    assert classifier.classify("Xin chào!")["label"] == "greeting"
    assert classifier.classify("nhắc tôi lúc 3 giờ")["label"] == "schedule_reminder"
    assert progress == sorted(progress) and progress[-1] == 0.5


def test_tfidf_rejects_disallowed_pickle(tmp_path):
    write_export(tmp_path, collections.OrderedDict(a=1))
    with pytest.raises(ValueError, match="disallowed collections.OrderedDict"):
        TfidfIntentClassifier.load(str(tmp_path))


def test_tfidf_rejects_changed_or_unapproved_export(export):
    config_path = export / "inference_config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps({**config, "model_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="checksum"):
        TfidfIntentClassifier.load(str(export))
    config_path.write_text(json.dumps({**config, "constraints_met": False}))
    with pytest.raises(ValueError, match="threshold"):
        TfidfIntentClassifier.load(str(export))


def test_ids_come_from_labels_and_threshold_sends_no_intent(client, service, ws_reply):
    # Model index 0 means something else in this model: the wire ID must follow the label.
    service.classifier = Mock(threshold=0.4)
    service.classifier.classify.return_value = {"id": 0, "label": "greeting", "score": 0.9}
    with client.websocket_connect("/ws/asr") as ws:
        ws.send_bytes(PCM)
        ws.send_text("END")
        assert ws_reply(ws)[0] == "2"
        service.classifier.classify.return_value = {"label": "greeting", "score": 0.3}
        ws.send_bytes(PCM)
        ws.send_text("END")
        assert ws_reply(ws)[0] == "-1"
    clips = client.get("/dashboard/recordings").json()
    assert clips[0]["result"]["intent"]["accepted"] is False
    assert clips[1]["result"]["intent"]["accepted"] is True


@pytest.fixture
def switcher(service, tmp_path, monkeypatch):
    (tmp_path / "inference_config.json").write_text("{}")
    settings = SimpleNamespace(
        intent_enabled=True, intent_model_id="hub/model", intent_tfidf_dir=str(tmp_path)
    )
    release = threading.Event()
    loaded = []

    def fake_load(name, threshold):
        def load(source, progress):
            progress(0.5, "Loading weights")
            assert release.wait(5)
            loaded.append(name)
            return Mock(threshold=threshold, classify=Mock(return_value={"label": "greeting"}))

        return load

    monkeypatch.setattr(IntentClassifier, "load", fake_load("bamibert", None))
    monkeypatch.setattr(TfidfIntentClassifier, "load", fake_load("tfidf_lr", 0.4))
    switcher = IntentSwitcher(settings, classifier=Mock(), active="bamibert")
    service.intents = switcher
    return SimpleNamespace(switcher=switcher, release=release, loaded=loaded)


def wait_until_idle(switcher):
    for _ in range(200):
        if switcher.status()["loading"] is None:
            return
        time.sleep(0.01)
    raise AssertionError("Intent model never finished loading")


def test_dashboard_switch_unloads_blocks_and_reports_progress(client, switcher):
    page = client.get("/dashboard/intent-model").json()
    assert page["active"] == "bamibert"
    assert [option["key"] for option in page["options"]] == ["bamibert", "tfidf_lr"]
    assert all(option["available"] for option in page["options"])

    post = client.post("/dashboard/intent-model", json={"model": "tfidf_lr"})
    assert post.status_code == 202
    for _ in range(200):
        status = client.get("/dashboard/intent-model").json()
        if status["progress"] == 47:
            break
        time.sleep(0.01)
    # The previous model is gone before the new one finishes loading.
    assert status == {**status, "active": None, "loading": "tfidf_lr", "stage": "Loading weights"}
    assert switcher.switcher.current() == (None, None)
    blocked = client.post("/dashboard/intent-model", json={"model": "bamibert"})
    assert blocked.status_code == 409

    switcher.release.set()
    wait_until_idle(switcher.switcher)
    status = client.get("/dashboard/intent-model").json()
    assert (status["active"], status["progress"], status["error"]) == ("tfidf_lr", 100, None)
    assert switcher.loaded == ["tfidf_lr"]
    classifier, key = switcher.switcher.current()
    assert key == "tfidf_lr" and classifier.threshold == 0.4
    classifier.classify.assert_called_once_with("Xin chào.")  # Warm-up.

    same = client.post("/dashboard/intent-model", json={"model": "tfidf_lr"})
    assert same.status_code == 200 and switcher.loaded == ["tfidf_lr"]


def test_failed_switch_reports_error_and_allows_retry(client, switcher, monkeypatch):
    monkeypatch.setattr(TfidfIntentClassifier, "load", Mock(side_effect=OSError("disk gone")))
    assert client.post("/dashboard/intent-model", json={"model": "tfidf_lr"}).status_code == 202
    wait_until_idle(switcher.switcher)
    status = client.get("/dashboard/intent-model").json()
    assert status["active"] is None
    assert status["error"] == "Could not load TF-IDF + Logistic Regression: disk gone"
    switcher.release.set()
    assert client.post("/dashboard/intent-model", json={"model": "bamibert"}).status_code == 202
    wait_until_idle(switcher.switcher)
    assert client.get("/dashboard/intent-model").json()["active"] == "bamibert"


def test_switch_requests_are_validated(client, switcher):
    url = "/dashboard/intent-model"
    assert client.post(url, json={"model": "gpt"}).status_code == 400
    assert client.post(url, content='{"model": "tfidf_lr"}').status_code == 415
    evil = client.post(url, json={"model": "tfidf_lr"}, headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403
    assert client.post(url, json={}).status_code == 400
    switcher.switcher.settings.intent_tfidf_dir = "/missing"
    assert client.post(url, json={"model": "tfidf_lr"}).status_code == 400
    assert switcher.loaded == []
