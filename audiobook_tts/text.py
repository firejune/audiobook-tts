# coding=utf-8
# Copyright 2026 The Audiobook-TTS Authors.
# SPDX-License-Identifier: Apache-2.0
"""
Text processing and robust line-aware sentence tokenization utilities for continuous audiobook generation.
Filters out markdown divider lines (---, ***), strips quotation marks, and bundles ultra-short exclamations
(e.g., '퍽! 퍽! 퍽! 퍽!') into natural, coherent narration units.
"""

import base64
import io
import re
from typing import List

import numpy as np
import soundfile as sf


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into distinct sentences optimized for audiobook pacing and narration flow.

    Key behaviors:
    1. Pure divider lines (---, ***, ===, ___) without words are completely dropped.
    2. All disruptive quotation marks are stripped to prevent acoustic hallucinations.
    3. Ultra-short fragments on the same line (e.g. '퍽! 퍽! 퍽! 퍽!', '아! 으악! 살려줘!')
       are bundled into a single natural narration unit rather than being chopped into sub-second tracks.
    4. Hanging Korean particles ('라고', '하고', '이라며') remain merged with dialogue.
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1. Normalize typographic curved quotes
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

    # 2. Filter out pure markdown / scene divider lines (---, ***, ===, ___, etc.)
    clean_lines = []
    for l in text.splitlines():
        l_str = l.strip()
        if not l_str:
            continue
        # Drop pure divider lines with no alphanumeric characters
        if re.match(r'^[-*=_~#\s]{2,}$', l_str) or not any(c.isalnum() for c in l_str):
            continue
        clean_lines.append(l_str)

    if not clean_lines:
        return []

    # 3. Sentence terminal punctuation pattern (period, exclamation, question mark)
    punct_pattern = re.compile(r'([.!?。！？…]+(?:\s+|$))')
    quote_clean = re.compile(r'["\'“”‘’「」『』`]')
    particles = ('라고', '하고', '이라며', '라며', '하며', '면서', '등', '은', '는', '이', '가', '을', '를')

    results: List[str] = []

    for line in clean_lines:
        # Strip all quotes from the line
        line = quote_clean.sub('', line).strip()
        if not line:
            continue

        # Split line by sentence-ending punctuation
        parts = punct_pattern.split(line)
        it = iter(parts)
        buffer = ""
        line_chunks = []
        for item in it:
            if not item:
                continue
            if punct_pattern.fullmatch(item):
                buffer += item.strip()
                if buffer.strip():
                    line_chunks.append(buffer.strip())
                buffer = ""
            else:
                buffer += item

        if buffer.strip():
            line_chunks.append(buffer.strip())

        # Merge hanging Korean particles within the line
        merged_chunks = []
        for s in line_chunks:
            s = s.strip()
            if not s:
                continue
            if merged_chunks and (
                not any(c.isalnum() for c in s) or
                any(s.startswith(p) for p in particles)
            ):
                merged_chunks[-1] = f"{merged_chunks[-1]} {s}".strip()
            else:
                merged_chunks.append(s)

        # Smart bundle consecutive short fragments (< 6 chars) on the same line
        # e.g., '퍽!', '퍽!', '퍽!', '퍽!' -> '퍽! 퍽! 퍽! 퍽!'
        bundled = []
        accum = ""
        for chunk in merged_chunks:
            if not accum:
                accum = chunk
            else:
                # If current accumulator is short (< 6 chars) or incoming chunk is ultra-short (< 5 chars),
                # keep them bundled in the same narration chunk!
                if len(accum) < 6 or len(chunk) < 5:
                    accum = f"{accum} {chunk}"
                else:
                    bundled.append(accum)
                    accum = chunk

        if accum:
            bundled.append(accum)

        for b in bundled:
            b_clean = re.sub(r'\s+', ' ', b).strip()
            if any(c.isalnum() for c in b_clean):
                results.append(b_clean)

    if results:
        return results

    fallback = quote_clean.sub('', text).strip()
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
