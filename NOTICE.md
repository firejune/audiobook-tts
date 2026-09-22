# Third-Party Notices and Attribution

## Upstream Base Models & Architectures

This project integrates and builds upon model architectures and components originating from:

### Qwen3-TTS
- **Source**: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- **Copyright**: Copyright (c) Alibaba Group.
- **License**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

Key parts of the core model definitions in `audiobook_tts/backends/qwen/` are adapted from the official Qwen3-TTS release under the terms of the Apache 2.0 license.

---

## Model Weights & Checkpoints

Model weights (such as `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` or tokenizer configurations) are downloaded dynamically at runtime from Hugging Face Hub. 

1. **No model weights or checkpoint binaries are committed to this repository or distributed in the Python package.**
2. Model checkpoints downloaded by users remain governed by their respective upstream licenses and acceptable use policies as published by Alibaba Cloud / QwenLM on the Hugging Face Model Hub.
3. Users are responsible for complying with the upstream terms of service and commercial usage guidelines associated with the respective checkpoint used.

---

## Software Dependencies

This project relies on open-source libraries including:
- **PyTorch & TorchAudio**: BSD-style license.
- **FastAPI & Uvicorn**: MIT License.
- **SoundFile**: BSD License.
- **Hugging Face Transformers & Accelerate**: Apache License 2.0.
