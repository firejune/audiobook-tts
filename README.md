<p align="center">
  <img src="https://raw.githubusercontent.com/firejune/audiobook-tts/main/assets/banner.svg" alt="audiobook-tts - Streaming Multi-Device Audiobook Studio" width="100%" />
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-38BDF8.svg?style=flat-square" alt="License: Apache-2.0" /></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3B82F6.svg?style=flat-square" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-2.6%20|%20CUDA%2013-10B981.svg?style=flat-square" alt="PyTorch CUDA 13" />
  <img src="https://img.shields.io/badge/Streaming-FastAPI%20SSE-A855F7.svg?style=flat-square" alt="FastAPI SSE" />
  <img src="https://img.shields.io/badge/Frontend-Zero--Build%20Studio-F59E0B.svg?style=flat-square" alt="Zero-Build Web Studio" />
</p>

**Streaming multi-device audiobook generator.** Plain literary manuscripts in, continuous high-fidelity audio chapters out. Satures heterogeneous GPU and CPU clusters simultaneously so listeners never wait for an entire chapter to generate.

---

## What you get

Plain text, articles, or novel chapters in; **instant, uninterrupted narration out**. The browser player begins speaking sentence 1 within seconds, while background worker engines generate upcoming paragraphs across all available accelerators in parallel.

| You have | You run | You get |
| --- | --- | --- |
| Manuscript, novel chapter, or news article | `audiobook-tts` | Zero-build Web Studio in your browser, continuous playback, auto-scrolling sentence queue |
| Multiple GPUs (`RTX 5070 Ti` + `RTX 4070 Ti`) | `audiobook-tts --devices cuda:1,cuda:0` | Concurrent sentence generation with dynamic load balancing |
| Heterogeneous rig (GPUs + Host CPU) | `audiobook-tts --devices cuda:1,cuda:0,cpu` | Full hardware saturation — every silicon core produces audio simultaneously |
| Natural emotion or voice acting requirements | Web Studio or `--instruct "Speak with suspense"` | Expressive prosody, whispering, cheerful storytelling, or dramatic pacing |

Everything runs locally or across your LAN: no telemetry, no cloud subscriptions, no external API keys required.

---

## What audiobook-tts is, and what it is not

`audiobook-tts` is built specifically for **long-form literature and continuous listening**:

- **It is a real-time streaming production studio, not a batch export script.** Traditional TTS demos force you to wait minutes for a 2,000-word chapter to complete before emitting a single sound. `audiobook-tts` splits narrative text into sentence units, dispatches them across an asynchronous worker pool, and streams audio to the client via Server-Sent Events (SSE). The listener hears the opening sentence almost immediately.
- **It is a heterogeneous hardware orchestrator.** Most deep learning engines bind to a single GPU, leaving secondary graphics cards or powerful multi-core host CPUs (e.g. Intel Core Ultra / AMD Ryzen 24-core) completely idle. `audiobook-tts` treats compute devices as a shared worker pool (`cuda:1`, `cuda:0`, `cpu`), saturating all silicon concurrently.
- **It is model-agnostic above the weights.** Today it links Alibaba's `Qwen3-TTS-12Hz` model family for 24kHz prosodic clarity. The engine layer is designed with pluggable backends (see [ROADMAP.md](ROADMAP.md)), allowing future integration of CosyVoice, ChatTTS, and local ONNX/GGUF models.

---

## Heterogeneous Worker Pool Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │              Input Manuscript / Chapter                │
               └───────────────────────────┬────────────────────────────┘
                                           │ Tokenize & Split
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │           Sentence Work Queue (Async FIFO)             │
               │   [Sentence 1]   [Sentence 2]   [Sentence 3]   [...]   │
               └───────┬───────────────────┬───────────────────┬────────┘
                       │                   │                   │
                       ▼                   ▼                   ▼
           ┌──────────────────────┐ ┌──────────────┐ ┌──────────────────┐
           │   Worker 0 (CUDA:1)  │ │Worker 1(CUDA)│ │  Worker 2 (CPU)  │
           │  RTX 5070 Ti (16GB)  │ │ RTX 4070 Ti  │ │ Core Ultra 7 265K│
           └──────────┬───────────┘ └──────┬───────┘ └─────────┬────────┘
                      │                    │                   │
                      └────────────────────┼───────────────────┘
                                           │ Push Completed Chunk
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │         FastAPI Server-Sent Events (SSE) Stream        │
               └───────────────────────────┬────────────────────────────┘
                                           │ Real-time Event Push
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │    Zero-Build Web Studio (Continuous Audio Engine)     │
               │  ▶ Playing Sentence 1  │  ⏳ Buffering 2, 3 in parallel │
               └────────────────────────────────────────────────────────┘
```

1. **Async Work Distribution**: Each physical worker continuously polls the task queue without global lock contention. As soon as Worker 0 finishes sentence 1, it immediately grabs sentence 4.
2. **Resilient Playback State Machine**: The client Web Studio maintains a strict sequential playback queue. If Worker 1 finishes sentence 3 before Worker 0 finishes sentence 2, sentence 3 buffers silently without interrupting narrative flow.
3. **Autoplay Watchdog**: An automated recovery watchdog monitors buffer transitions, resuming playback without audio stutter or frozen UI states.

---

## Quickstart

### Prerequisites
- Python 3.10+
- PyTorch 2.4+ (PyTorch 2.6+ with CUDA 13 recommended for RTX 50-series Blackwell GPUs)

### Installation

```bash
# Clone the repository
git clone https://github.com/firejune/audiobook-tts.git
cd audiobook-tts

# Install in editable mode
pip install -e .
```

### Running the Studio

```bash
# Launch with all available hardware accelerators (e.g. Dual GPUs + CPU)
audiobook-tts --devices cuda:1,cuda:0,cpu --port 8000

# Or launch with a single GPU
audiobook-tts --devices cuda:0 --port 8000

# Launch with Apple Silicon (MPS)
audiobook-tts --devices mps --port 8000
```

Open your browser at **`http://localhost:8000`** to access the Audiobook Studio.

---

## CLI Reference

```text
usage: audiobook-tts [-h] [-c CHECKPOINT] [--device DEVICE] [--devices DEVICES]
                     [--dtype {bfloat16,bf16,float16,fp16,float32,fp32}]
                     [--ip IP] [--port PORT]
                     [checkpoint_pos]

Audiobook-TTS Streaming Studio Server

positional arguments:
  checkpoint_pos        Model checkpoint path or Hugging Face repository

options:
  -h, --help            show this help message and exit
  -c, --checkpoint      Model checkpoint path or Hugging Face repository
  --device DEVICE       Single compute device (e.g. mps, cuda:0, cpu)
  --devices DEVICES     Comma-separated compute devices for worker pool
                        (e.g. cuda:1,cuda:0,cpu)
  --dtype               Computation precision (default: bfloat16)
  --ip IP               Host IP address (default: 0.0.0.0)
  --port PORT           Port to listen on (default: 8000)
```

---

## Web Studio Features

- **Zero-Build Portability**: Single standalone `web/index.html` file using pure Vanilla JS/CSS. No Node.js build step, no Webpack/Vite bundler required.
- **Hardware Telemetry Badges**: Live indicators displaying active neural model checkpoint, compute devices assigned to the worker pool, and audio fidelity.
- **Sentence-Level Visual Queue**: Track cards show generation status (`Pending`, `Generating`, `Buffering`, `Playing`, `Completed`) and which specific accelerator generated that sentence.
- **Autonomous Playback Recovery**: Unlocks HTML5 audio policies seamlessly; resumes immediately when buffer underruns clear.
- **Voice Preset Selector**: Choose between standard voices (`Vivian`, `Serena`, `Ryan`, `Uncle`) and customize emotional prosody on the fly.

---

## Third-Party Notices & Licensing

- `audiobook-tts` code, architecture, server, and Web Studio are licensed under the **[Apache License 2.0](LICENSE)**.
- Base neural model definitions in `qwen_tts/` are derived from [Alibaba QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0).
- Model weights are downloaded dynamically from Hugging Face Hub and are subject to the upstream acceptable use policy. See [NOTICE.md](NOTICE.md) for full attribution details.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for future milestones, including:
- Dialogue & Multi-Speaker Tagging (automated actor casting).
- EPUB, PDF, and Markdown book file ingestion.
- One-click export to standard **M4B (AAC)** with chapter markers and cover art.
- Pluggable backend adapters for CosyVoice and F5-TTS.
