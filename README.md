# ESP32 Vietnamese ASR server

Stream push-to-talk audio from an ESP32-S3 to a Python host on the same Wi-Fi network. Send `END` to receive filtered audio and the predicted intent ID over the open WebSocket. The dashboard pairs both audio versions with the Vietnamese transcript. The FastAPI process runs Qwen ASR and the fine-tuned BamiBERT classifier locally.

## Installation

Use Python 3.12 on the inference host. An NVIDIA CUDA GPU is preferable for interactive use. CPU inference uses FP32 and may take considerably longer. Run commands from the project root.

```sh
python -m venv .venv
source .venv/bin/activate
```

On NVIDIA hosts, check the installed driver with `nvidia-smi`. Select the PyTorch pip installation command for your OS and a CUDA runtime supported by that driver. Run it in the activated environment before installing the server. Install matching PyTorch-family wheels if your platform requires them. [PyTorch installation selector](https://pytorch.org/get-started/locally/)

```sh
pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
cp .env.example .env
```

## Checkpoint and startup

The first startup resolves `duclvQ/qwen3-asr-1.7b-vimd-full` and downloads root Safetensors weights plus tokenizer and processor assets into the Hugging Face cache. The fine-tune contains BF16 weights. The server loads the whole checkpoint on one device with `Qwen3ASRModel.from_pretrained()`.

On every startup, the server asks the Hub for the newest commit of each Hub-hosted model. If there is a new commit, it downloads only the changed files. Otherwise it reuses the cache. The log shows `Using <repo> at commit <hash>` for each model. If the Hub is unreachable or returns a server error, startup logs a warning and loads the cached copy. Authentication and missing-repository errors still stop startup. A model given as a local directory is never checked. Updates happen only at startup, so restart the server to pick up a new commit. If a new classifier commit changes the label order, update the ESP32 `IntentId` enum to match. [ViMD model card](https://huggingface.co/duclvQ/qwen3-asr-1.7b-vimd-full) · [Qwen backend](https://github.com/QwenLM/Qwen3-ASR)

To prefetch before deployment, run either command. The script prints the local snapshot path.

```sh
python scripts/download_model.py
python scripts/download_model.py --cache-dir /data/hf-cache
```

For a custom cache, set `HF_HUB_CACHE=/data/hf-cache` in the launch environment or set `MODEL_ID` to the printed snapshot path. For disconnected operation, set `HF_HUB_OFFLINE=1` before launch. This skips the update check. Preserve the entire cache, including blobs referenced by snapshot symlinks, when moving it between machines. Keep caches outside the source tree.

Startup also loads the [Minimate intent classifier](https://huggingface.co/nguen2929/minimate_intent_classifier/) and its tokenizer on CPU. To prepare for offline operation, start the server online once or prefetch the classifier with `hf download nguen2929/minimate_intent_classifier`. For a private checkpoint, authenticate with `hf auth login` before the first download. The classifier is pinned to one commit, so its tokenizer and weights always come from the same revision. You can also set `INTENT_MODEL_ID` to a local Transformers export containing the tokenizer, trained head, and `config.json` with the intent names in `id2label`. An enabled classifier must load before the server is ready. Set `INTENT_ENABLED=false` for transcription-only operation.

A second intent model is available: a TF-IDF + logistic regression pipeline trained by another team member, with a validated confidence threshold of 0.40. Place its `intent_pipeline.joblib` and `inference_config.json` in `INTENT_TFIDF_DIR`, which defaults to `models/minimate_synthetic_v1` (gitignored). The server checks the SHA-256 in `inference_config.json` before loading, and loads the pickle through an allowlist that refuses any class except the 8 sklearn, numpy, and joblib classes the pipeline uses. It needs `scikit-learn` 1.9.x. A version mismatch logs one warning. `INTENT_DEFAULT_MODEL` picks the model loaded at startup.

```sh
python3 server.py
```

Run this command from the project root in your activated virtual environment. The server reads its bind settings from `.env` and starts one Uvicorn worker.

Before serving requests, startup loads the models and runs one warm-up using one second of silence for ASR and a short Vietnamese phrase for the enabled classifier. Both models stay in memory between requests, including while the server is idle. A load or warm-up failure stops startup. Shutdown drains active model work before releasing the weights.

| Endpoint | Response |
| --- | --- |
| `GET /` | Service name, sample rate, WebSocket path |
| `GET /health` | Process health and configured model identity |
| `GET /ready` | HTTP 200 when the model can accept work, otherwise HTTP 503 when the endpoint is reachable |
| `GET /dashboard` | Local recording playback, transcripts, and ESP32 device log |
| `WS /ws/asr` | One or more push-to-talk recordings per connection |

Open [the recording dashboard](http://127.0.0.1:8000/dashboard) on the server laptop. Each recording shows the transcript and intent as they become available. Play or download the **Original** WAV to hear what arrived from the ESP32 and the **Filtered** WAV to hear what ASR received. The dashboard keeps the last 10 valid recordings in memory and clears them on restart. Access is limited to localhost.

The **Intent model** dropdown at the top of the dashboard chooses which model classifies transcripts: **BamiBERT** or **TF-IDF + Logistic Regression**. Only the chosen model stays in memory. Switching unloads the current model first, then loads the new one in the background. The status shows a percentage and the current step, for example `Loading BamiBERT… 66% · Loading weights`. The dropdown is disabled until the load finishes, and the server refuses another switch with HTTP 409 until then. Transcripts that arrive during a load get no intent (`-1`). If a load fails, the error is shown, no model is loaded, and you can choose again. The choice lasts until restart, which loads `INTENT_DEFAULT_MODEL` again. Each recording shows which model produced its intent.

The **Device log** section shows the last 200 log lines the ESP32 sent over its WebSocket, with the receive time and the device IP. Each line starts with the device's uptime. Use it when the board runs on external power with no USB serial. The server also writes each line to its own log at INFO. Lines are kept in memory only and cleared on restart.

## Configuration

Process environment values override `.env`. Restart after changes.

| Setting | Default | Meaning |
| --- | --- | --- |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | Server bind address and port |
| `MODEL_ID` | `duclvQ/qwen3-asr-1.7b-vimd-full` | Hub ID or local copy of its original root checkpoint |
| `MODEL_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N` |
| `MODEL_DTYPE` | `auto` | CUDA uses native BF16 when supported, otherwise FP16. Explicit CUDA choices are `bf16` and `fp16`. CPU defaults to FP32 and can explicitly use `bf16`. |
| `INTENT_ENABLED` | `true` | Classify nonempty transcriptions. Set to `false` for transcription only. |
| `INTENT_MODEL_ID` | `nguen2929/minimate_intent_classifier` | Hub ID or local BamiBERT classifier export |
| `INTENT_DEFAULT_MODEL` | `bamibert` | Intent model loaded at startup: `bamibert` or `tfidf_lr` |
| `INTENT_TFIDF_DIR` | `models/minimate_synthetic_v1` | Folder holding the TF-IDF `intent_pipeline.joblib` and `inference_config.json` |
| `MAX_NEW_TOKENS` | `256` | Maximum generated tokens per utterance |
| `SAMPLE_RATE`, `ASR_LANGUAGE` | `16000`, `Vietnamese` | Fixed ESP32 protocol settings |
| `MIN_AUDIO_SECONDS`, `MAX_AUDIO_SECONDS` | `0.4`, `20` | Inclusive limits, checked in whole samples |
| `MAX_CONCURRENT_INFERENCES` | `1` | One active model call per process |
| `LOG_LEVEL` | `INFO` | Application log threshold |
| `HIGHPASS_HZ` | `100` | Remove audio below this frequency before ASR, from 0 to 300. `0` disables the filter |

## ESP32 connection and protocol

For a serial-controlled ESP32-S3 and INMP441 client, see the [Arduino sketch and wiring guide](minimate-esp32s3-client/README.md).

Configure the ESP32-S3 and INMP441 capture path to produce **16 kHz, signed little-endian PCM16, mono**. If I2S capture uses wider slots, select the active microphone channel and convert its samples to PCM16 on the device. Connect to `ws://SERVER_IP:8000/ws/asr` using the inference host’s LAN IP. The receiver cannot infer a sample rate or channel count from raw PCM.

1. During recording, send raw PCM in binary WebSocket messages. Typical chunks are 1,600–3,200 bytes. One second occupies 32,000 bytes. Omit file headers.
2. On release, send the exact text message `END`. Wait for a reply before starting another recording.
3. Read the filtered PCM16 as binary chunks of up to 1,600 bytes. The full reply has the same number of audio bytes as the request. A final text message carries the intent ID and marks the end of the reply. For example, `2` means greeting. `-1` means the server could not provide an intent. Invalid recordings or failed transcription return only the error text response. [ESP32 client](minimate-esp32s3-client/README.md)

Each connection buffers its own recording. At `END`, the server applies a short median-based filter for sharp digital clicks. It then removes everything below `HIGHPASS_HZ`, reaching full level at 1.25 times that frequency. This takes out DC offset, rumble, and power-supply noise picked up by the microphone. The 300–4000 Hz speech band is unchanged. Both filters are zero-phase and keep the exact sample count. The server converts the filtered PCM to normalized NumPy Float32 for transcription. Broadband hiss and room noise are not removed. Inference runs in a worker thread under a shared lock. Audio stays in memory. Only outer whitespace and the exact leading `language Vietnamese<asr_text>` marker are trimmed from the transcript.

The same worker passes the cleaned transcript to the selected intent model. The reply's intent ID comes from the predicted label name, in the order of the ESP32 `IntentId` enum, so both models send the same ID for the same intent. If the TF-IDF model's confidence is below its threshold, the reply is `-1`, and the dashboard marks the prediction as below threshold. The dashboard shows the label, score, and model with the transcript and audio. It shows no intent when classification is disabled, the transcript is empty, or classification fails. Classifier errors are logged on the server.

An odd-length frame or overflowing recording receives `-1` immediately. Send the terminating `END` to clear the discarded recording before starting another, or reconnect to clear the connection's buffer. A text message that starts with `LOG ` carries one device log line. The server stores the first 500 characters for the dashboard and sends no reply. Log lines can arrive at any time, including during a recording, and leave the recording intact. Other unknown text messages are also ignored. Disconnecting discards the partial utterance.

## WAV client and benchmark

Use an uncompressed 16 kHz mono PCM16 WAV. The client extracts raw frames and sends them as fast as possible by default. The WAV client and benchmark request the `asr-json` WebSocket subprotocol for transcripts, scores, timings, and detailed errors.

```sh
python scripts/test_client.py sample.wav
python scripts/test_client.py sample.wav --filtered-wav filtered.wav
python scripts/test_client.py sample.wav --server ws://127.0.0.1:8000/ws/asr --chunk-ms 100 --realtime
python scripts/benchmark.py sample.wav
python scripts/benchmark.py clips/ --csv results.csv
```

The server reports ASR model time in seconds after acquiring the inference lock. CUDA measurements synchronize the selected device. The client shows the predicted intent when available and measures elapsed time from sending `END` to receiving the reply. The benchmark measures request time from the first audio send through the reply, including intent classification, network transfer, and queueing. Connection setup and WAV decoding are outside that timer.

Real-time factor is model time divided by audio duration. Values below one indicate faster-than-real-time inference. GPU allocated and peak bytes are process allocator measurements on the selected device. Peak includes resident weights and temporary inference allocations. CPU results use JSON null for GPU metrics. Measure with representative recordings after the server is ready.

To evaluate accuracy, install the dev requirements. Supply a UTF-8 JSON object mapping each WAV path relative to the input directory to its reference transcript, e.g. `{"north/001.wav":"Xin chào"}`. For one file use its basename. The script reports corpus WER and CER separately from timing rows, using jiwer defaults with case and punctuation preserved.

```sh
python scripts/benchmark.py clips/ --references references.json --csv results.csv
```

## Verification

```sh
pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Tests substitute the model during lifespan and exercise the real WebSocket handler. The TCP loopback test starts Uvicorn with the fake model and skips if the OS denies sockets. To verify actual hardware, start the server, check `/ready`, submit a known Vietnamese WAV, and inspect its transcript, device and timing.

## Operations and limitations

- **CUDA unavailable or unexpected CPU selection.** Check driver visibility and the PyTorch CUDA probe above. Set `MODEL_DEVICE=cuda:0` to require a GPU. An unsupported explicit device or precision fails startup.
- **Out of memory.** Allow VRAM for weights, model activations, token generation and the CUDA runtime. Check for duplicate workers and other GPU processes. Reduce the recording limit or token budget if inference exceeds available memory. Initialization failure aborts startup. A failed request logs its exception and returns an error on the same connection. The server keeps its configured device.
- **CPU memory error while loading Qwen.** Set `MODEL_DEVICE=cpu` and `MODEL_DTYPE=bf16` in `.env` to load the original BF16 weights and roughly halve Qwen's weight memory versus FP32. CPU BF16 speed and operator support depend on your processor and PyTorch build. Startup runs warm-up with the selected dtype.
- **Offline load failure.** Confirm the cache directory and file permissions. Repeat the download online to populate missing shards, configuration and processor assets before enabling offline mode.
- **Distorted or unusually short text.** Check channel selection, endianness, I2S slot conversion, sample rate and token budget. The server transcribes complete utterances. It provides no interim text or voice activity detection. Its only noise reduction is the click filter and the low-frequency high-pass. Simultaneous clients share model capacity, so queues add latency.
- **ESP32 returns unrelated words.** Set `LOG_LEVEL=DEBUG` and look for `PCM levels` after speaking. A zero `signal_rms` means the captured samples were flat. Check that the INMP441 L/R wiring matches `MIC_SLOT`. A large `clipped` count points to a saturated or misdecoded signal. Compare with the same phrase sent through the WAV client to check whether the problem follows ESP32 capture.
- **LAN access and privacy.** Bind to a trusted LAN interface and restrict the port to intended devices. Anyone with network access can send recordings and consume GPU time. Plain WebSocket carries audio and transcripts without encryption. INFO logs include transcripts. Set `LOG_LEVEL=WARNING` to suppress normal transcript logs and restrict access to retained logs.

## Licensing

The server code uses the [MIT license](LICENSE). The fine-tune’s model card declares Apache-2.0. Its training dataset declares CC BY-NC-ND 4.0. Review the original terms and the model author’s commercial-use caveat for your deployment. Weights and dataset are distributed separately. [Model card](https://huggingface.co/duclvQ/qwen3-asr-1.7b-vimd-full) · [ViMD dataset card](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset)
