# coding=utf-8
# Copyright 2026 The Audiobook-TTS Authors.
# SPDX-License-Identifier: Apache-2.0
"""
Text processing and robust sentence tokenization utilities for continuous audiobook generation.
Completely strips disruptive quotation marks while preserving sentence boundaries and dialogue flow.
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
    Preserves dialogue context, particles, and exclamation/question marks,
    while COMPLETELY stripping all quotation marks to prevent TTS token hallucination
    and language glitching.
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1. Normalize typographic curved quotes to standard quotes for boundary detection
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

    # Split lines first to respect paragraph breaks
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    raw_results = []

    # Pattern: sentence ending punctuation (. ! ? 。 ！ ？ …)
    # followed by optional quotes, then space or end of line.
    punct_pattern = re.compile(r'([.!?。！？…]+["\'\)\]]*(?:\s+|$))')

    for line in lines:
        parts = punct_pattern.split(line)
        it = iter(parts)
        buffer = ""
        for item in it:
            if not item:
                continue
            if punct_pattern.fullmatch(item):
                buffer += item.strip()
                if buffer.strip():
                    raw_results.append(buffer.strip())
                buffer = ""
            else:
                buffer += item

        if buffer.strip():
            raw_results.append(buffer.strip())

    # 2. Merge hanging Korean particles or dialogue continuation
    particles = ('라고', '하고', '이라며', '라며', '하며', '면서', '등', '은', '는', '이', '가', '을', '를')
    merged = []
    for s in raw_results:
        s = s.strip()
        if not s:
            continue

        if merged and (
            not any(c.isalnum() for c in s) or
            any(s.startswith(p) for p in particles)
        ):
            merged[-1] = f"{merged[-1]} {s}".strip()
        else:
            merged.append(s)

    # 3. Completely strip all quotation marks to guarantee pristine TTS acoustic output
    quote_clean_pattern = re.compile(r'["\'“”‘’「」『』`]')
    cleaned = []
    for s in merged:
        # Strip all forms of quotes
        s_clean = quote_clean_pattern.sub('', s).strip()
        # Collapse multiple spaces into one
        s_clean = re.sub(r'\s+', ' ', s_clean).strip()

        # Must contain at least one alphanumeric character
        if any(c.isalnum() for c in s_clean):
            cleaned.append(s_clean)

    if cleaned:
        return cleaned

    fallback = quote_clean_pattern.sub('', text).strip()
    return [fallback] if fallback else [text]


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
