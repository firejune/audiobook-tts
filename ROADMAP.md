# Roadmap to 1.0

Rough, and deliberately so. No dates, no feature list.

## What this is for

**Uninterrupted, high-fidelity audiobook generation from long-form literature without human intervention.** That is the end state. Plain novel or manuscript text enters; continuous, multi-character narration emerges in real time.

Qwen3-TTS is **the first engine this supports, not what this project is.** Today it serves as the baseline voice model because its 12Hz representation and zero-shot voice cloning are capable of expressive prosody. But the core value of `audiobook-tts` lies above the weights:
1. Orchestrating heterogeneous compute clusters to eliminate queuing latency.
2. Ingesting raw literary text and automatically attributing dialogue to distinct character voices.
3. Streaming playback instantaneously while background workers generate upcoming paragraphs ahead of the listener.

---

## The Core Bets

### 1. Heterogeneous Hardware Saturation
Single-device inference for long-form books is fundamentally flawed: discrete GPUs sit idle between chapters, and powerful multi-core host CPUs remain completely unutilized.
- **Where we are:** A concurrent worker pool distributing discrete sentences across multiple GPUs (`cuda:1`, `cuda:0`) and host CPU (`cpu`) simultaneously.
- **Where we are going:** Dynamic token-length aware scheduling. Shorter narrative bridges are routed to CPU threads or secondary GPUs, while emotionally dense or complex dialogue passages receive primary accelerator priority.

### 2. Dialogue Segmentation & Automated Voice Casting
Manual SSML markup and hand-annotated voice tags ruin the reading experience. Literature exists in plain text.
- **Where we are going:** Zero-shot dialogue and speaker attribution. An LLM pre-pass analyzes quotation marks, dialogue tags ("she murmured", "he shouted from across the room"), tracks character identities across chapters, and automatically assigns designated voice presets.

### 3. Publication Packaging (Lossless M4B & Chapter Caching)
Listening live in a web browser is only the authoring phase. Readers consume audiobooks on mobile devices, offline, and in dedicated players.
- **Where we are going:** Single-click export to standard **M4B (AAC) / MP3** with embedded cover art, chapter markers, and sentence timestamps extracted directly from the generation pipeline.

### 4. Pluggable Synthesis Backends
Model architectures evolve rapidly. Binding an entire application lifecycle to a single proprietary or open model guarantees obsolescence.
- **Where we are going:** A unified engine contract (`TTSBackend`). Pluggable backend adapters for CosyVoice, ChatTTS, F5-TTS, and local GGUF/ONNX quantizations, with identical streaming SSE semantics.

---

## Milestones

| Milestone | Focus | Status |
| --- | --- | --- |
| **0.1.0** | Heterogeneous Worker Pool (`cuda:x` + `cpu`), Dynamic SSE streaming, Zero-build Web Studio | **Shipped** |
| **0.2.0** | Dialogue & Character Voice Registry (automatic speaker mapping) | Planned |
| **0.3.0** | Document Ingestion Engine (EPUB, PDF, Markdown, TXT) with chapter boundary detection | Planned |
| **0.4.0** | M4B / MP3 Container Packaging with ID3/M4A metadata & chapter marks | Planned |
| **0.5.0** | Pluggable Engine Abstraction (CosyVoice & F5-TTS backend adapters) | Planned |
| **1.0.0** | Autonomous end-to-end book compilation with verifiable voice consistency | Planned |
