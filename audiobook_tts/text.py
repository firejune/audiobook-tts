# coding=utf-8
# Copyright 2026 The Audiobook-TTS Authors.
# SPDX-License-Identifier: Apache-2.0
"""
Text processing and robust sentence tokenization utilities for continuous audiobook generation.
Safeguards quotation integrity and dialogue context to prevent TTS token hallucination.
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
    Preserves quotation pairs, dialogue attribution, exclamation/question marks,
    and trailing particles (e.g., '"안 돼!"라고 소리쳤다') to avoid orphan quotes
    that trigger TTS hallucination or language glitching.
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1. Normalize typographic curved quotes to standard ASCII quotes
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

    # Split lines first to preserve explicit paragraph structure
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    raw_results = []

    # Pattern: match sentence-terminal punctuation (. ! ? 。 ！ ？ …)
    # followed by optional quotes/brackets, THEN followed by whitespace or end of line.
    # Note: If a quote is immediately followed by a particle (no space), e.g. "안 돼!"라고,
    # it is NOT treated as a sentence boundary.
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

    # 2. Post-processing: Merge hanging Korean particles or dialogue continuation
    particles = ('라고', '하고', '이라며', '라며', '하며', '면서', '등', '은', '는', '이', '가', '을', '를')
    merged = []
    for s in raw_results:
        s = s.strip()
        if not s:
            continue

        # If chunk is just punctuation or starts with a hanging particle, merge into previous chunk
        if merged and (
            not any(c.isalnum() for c in s) or
            any(s.startswith(p) for p in particles)
        ):
            merged[-1] = f"{merged[-1]} {s}".strip()
        else:
            merged.append(s)

    # 3. Balancing & sanitizing quotes:
    # Ensure no single orphaned quote ruins the neural acoustic model's attention
    cleaned = []
    for s in merged:
        # Strip chunks containing zero alphanumeric characters
        if not any(c.isalnum() for c in s):
            continue

        # If odd number of double quotes, balance them cleanly
        q_count = s.count('"')
        if q_count % 2 != 0:
            if s.startswith('"') and not s.endswith('"'):
                s = s + '"'
            elif s.endswith('"') and not s.startswith('"'):
                s = '"' + s

        cleaned.append(s)

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
