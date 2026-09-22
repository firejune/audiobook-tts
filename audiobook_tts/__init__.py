# coding=utf-8
# Copyright 2026 The Audiobook-TTS Authors.
# SPDX-License-Identifier: Apache-2.0
"""
Audiobook-TTS: Streaming Heterogeneous Multi-Device Audiobook Generator.
"""

__version__ = "0.1.0"

from .pool import WorkerEngine, WorkerPool
from .text import split_into_sentences, wav_to_base64_data_url

__all__ = [
    "WorkerEngine",
    "WorkerPool",
    "split_into_sentences",
    "wav_to_base64_data_url",
]
