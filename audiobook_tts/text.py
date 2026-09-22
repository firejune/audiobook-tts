# coding=utf-8
# Copyright 2026 The Audiobook-TTS Authors.
# SPDX-License-Identifier: Apache-2.0
"""
Text processing and sentence tokenization utilities for continuous audiobook generation.
"""

import base64
import io
import re
from typing import List

import numpy as np
import soundfile as sf


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into distinct sentences suitable for audiobook narrative pacing.
    Preserves quotes, question marks, exclamation marks, and dialogue structure.
    """
    text = (text or "").strip()
    if not text:
        return []

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    raw_results = []
    pattern = re.compile(r"([^.!?。！？\n]+[.!?。！？]+)")
    for line in lines:
        matched = pattern.findall(line)
        rem = pattern.sub("", line).strip()
        if matched:
            for m in matched:
                if m.strip():
                    raw_results.append(m.strip())
            if rem:
                raw_results.append(rem)
        else:
            raw_results.append(line)

    cleaned = []
    for r in raw_results:
        s = r.strip()
        if any(c.isalnum() for c in s):
            cleaned.append(s)
        elif cleaned:
            cleaned[-1] = f"{cleaned[-1]} {s}".strip()

    return cleaned if cleaned else [text]


def wav_to_base64_data_url(wav: np.ndarray, sr: int) -> str:
    """
    Encode float numpy waveform to standard 16-bit PCM WAV base64 data URL.
    Ensures universal cross-browser HTML5 Audio playback compatibility.
    """
    buf = io.BytesIO()
    wav = np.asarray(wav, dtype=np.float32)
    max_val = np.max(np.abs(wav))
    if max_val > 1.0:
        wav = wav / max_val
    wav_int16 = (wav * 32767.0).astype(np.int16)
    sf.write(buf, wav_int16, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"
