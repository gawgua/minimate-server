NO_INTENT = -1
# Wire intent IDs, in the ESP32 IntentId order. Models map their label names onto these IDs,
# so every model sends the same ID for the same intent.
INTENT_LABELS = (
    "check_focus_session",
    "focus_session_control",
    "greeting",
    "open_question",
    "out_of_scope",
    "repeat_response",
    "schedule_reminder",
)
INTENT_IDS = {label: index for index, label in enumerate(INTENT_LABELS)}
DIAGNOSTIC_SUBPROTOCOL = "asr-json"
# Text frames starting with this prefix carry a device log line, not a command.
LOG_PREFIX = "LOG "
MAX_LOG_CHARS = 500


class Recording:
    def __init__(self, min_samples: int, max_samples: int):
        self.minimum = min_samples * 2
        self.maximum = max_samples * 2
        self.buffer = bytearray()
        self.discard = False

    def append(self, chunk: bytes):
        if self.discard:
            return
        if len(chunk) % 2:
            error = "PCM16 requires even byte count"
        elif len(self.buffer) + len(chunk) > self.maximum:
            error = "Audio too long"
        else:
            self.buffer.extend(chunk)
            return
        self.buffer.clear()
        self.discard = True
        raise ValueError(error)

    def finish(self) -> bytes | None:
        if self.discard:
            self.discard = False
            return None
        if len(self.buffer) < self.minimum:
            self.buffer.clear()
            raise ValueError("Audio too short")
        pcm = bytes(self.buffer)
        self.buffer.clear()
        return pcm
