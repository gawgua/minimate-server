import asyncio
import logging
import re
import time

import numpy as np

from .checkpoint import checkpoint
from .config import RATE
from .intent_models import MODEL_NAMES, IntentSwitcher
from .protocol import INTENT_IDS, NO_INTENT

log = logging.getLogger(__name__)


def clean_transcript(text: str) -> str:
    return text.strip().removeprefix("language Vietnamese<asr_text>").strip()


def select_device(torch, device: str, dtype: str):
    cuda = torch.cuda
    if device == "auto":
        device = "cuda:0" if cuda.is_available() else "cpu"
    if device == "cpu":
        if dtype == "bf16":
            log.info("Using CPU BF16 inference to reduce model memory")
            return device, torch.bfloat16
        if dtype not in {"auto", "fp32"}:
            raise ValueError(f"Unsupported CPU dtype: {dtype}. Use auto, fp32, or bf16.")
        log.warning("CPU FP32 inference may be slow")
        return device, torch.float32
    if not re.fullmatch(r"cuda(?::[0-9]+)?", device) or not cuda.is_available():
        raise ValueError(f"Unavailable model device: {device}")
    index = int(device.partition(":")[2] or 0)
    if index >= cuda.device_count():
        raise ValueError(f"Unavailable CUDA index: {index}")
    with cuda.device(index):
        bf16 = cuda.is_bf16_supported(including_emulation=False)
    if dtype == "auto":
        dtype = "bf16" if bf16 else "fp16"
    if dtype == "bf16" and bf16:
        precision = torch.bfloat16
    elif dtype == "fp16":
        precision = torch.float16
    else:
        raise ValueError(f"Unsupported CUDA dtype: {dtype}")
    return f"cuda:{index}", precision


class ASRService:
    def __init__(self, model, torch, device, classifier=None, intents=None):
        self.model = model
        self.intents = intents or IntentSwitcher(classifier=classifier, active="bamibert")
        self.torch = torch
        self.device = device
        self.ready = True
        self.lock = asyncio.Lock()
        self.worker = None

    @property
    def classifier(self):
        return self.intents.current()[0]

    @classifier.setter
    def classifier(self, value):
        self.intents.set(value)

    @classmethod
    def load(cls, settings):
        import torch
        from qwen_asr import Qwen3ASRModel

        device, dtype = select_device(torch, settings.device, settings.dtype)
        intents = IntentSwitcher(settings)
        if settings.intent_enabled:
            key = settings.intent_default_model
            log.info("Loading intent classifier %s on CPU", MODEL_NAMES[key])
            try:
                intents.load_now(key)
            except Exception as exc:
                raise RuntimeError(f"Could not load intent classifier {MODEL_NAMES[key]}") from exc
        log.info("Loading %s on %s with %s", settings.model_id, device, dtype)
        model = Qwen3ASRModel.from_pretrained(
            str(checkpoint(settings.model_id)),
            device_map=device,
            dtype=dtype,
            max_inference_batch_size=1,
            max_new_tokens=settings.tokens,
            use_safetensors=True,
        )
        service = cls(model, torch, device, intents=intents)
        service._warm_up()
        return service

    def _recognize(self, samples):
        return self.model.transcribe(audio=(samples, RATE), language="Vietnamese")

    def _warm_up(self):
        with self.torch.inference_mode():
            log.info("Warming up Qwen ASR on %s", self.device)
            self._recognize(np.zeros(RATE, dtype=np.float32))
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize(self.device)
        # IntentSwitcher warms up each intent model as it loads.
        log.info("Inference models warmed up")

    def _infer(self, samples):
        cuda = self.torch.cuda if self.device.startswith("cuda") else None
        with self.torch.inference_mode():
            if cuda:
                cuda.synchronize(self.device)
                cuda.reset_peak_memory_stats(self.device)
            log.info("Inference started")
            start = time.perf_counter()
            try:
                result = self._recognize(samples)
            except Exception as exc:
                if cuda and isinstance(exc, self.torch.OutOfMemoryError):
                    log.error(
                        "CUDA out of memory on %s. Check available VRAM and recording length.",
                        self.device,
                    )
                raise
            if cuda:
                cuda.synchronize(self.device)
            elapsed = time.perf_counter() - start
            text = clean_transcript(result[0].text)
            log.info("Inference completed: %.3fs", elapsed)
            log.info("Transcription: %s", text)
            response = {
                "text": text,
                "device": self.device,
                "inference_seconds": elapsed,
                "gpu_allocated_bytes": cuda.memory_allocated(self.device) if cuda else None,
                "gpu_peak_bytes": cuda.max_memory_allocated(self.device) if cuda else None,
                "intent": None,
                "intent_id": NO_INTENT,
            }
            # Snapshot once: a dashboard switch may unload the model while this runs.
            classifier, model_key = self.intents.current()
            if text and classifier is not None:
                try:
                    prediction = classifier.classify(text)
                    threshold = getattr(classifier, "threshold", None)
                    accepted = threshold is None or prediction["score"] >= threshold
                    # IDs come from label names, so every model sends the ESP32 the same IDs.
                    if accepted:
                        response["intent_id"] = INTENT_IDS.get(prediction["label"], NO_INTENT)
                    response["intent"] = {
                        "label": prediction["label"],
                        "score": prediction["score"],
                        "model": model_key,
                        "accepted": accepted,
                    }
                except Exception:
                    log.exception("Intent classification failed")
            elif text and self.intents.loading:
                log.info("No intent: %s is loading", MODEL_NAMES[self.intents.loading])
            return response

    async def transcribe(self, samples):
        await self.lock.acquire()
        if not self.ready:
            self.lock.release()
            raise RuntimeError("Model unavailable")
        # The worker owns the lock until completion, even if its client disconnects.
        task = asyncio.create_task(asyncio.to_thread(self._infer, samples))
        self.worker = task
        task.add_done_callback(self._finished)
        return await asyncio.shield(task)

    def _finished(self, task):
        self.worker = None
        self.lock.release()
        if not task.cancelled():
            task.exception()  # Observe errors even when the original waiter disconnected.

    async def close(self):
        self.ready = False
        async with self.lock:
            self.model = None
            self.intents.close()
            if self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()
