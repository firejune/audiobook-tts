# coding=utf-8
# Copyright 2026 The Audiobook-TTS Authors.
# SPDX-License-Identifier: Apache-2.0
"""
Audiobook-TTS: Streaming Heterogeneous Multi-Device Audiobook Generator.
"""

__version__ = "0.1.0"

from .text import (
    is_audiobook_markdown,
    parse_audiobook_script,
    parse_frontmatter,
    split_into_sentences,
    wav_to_base64_data_url,
)

__all__ = [
    "WorkerEngine",
    "WorkerPool",
    "is_audiobook_markdown",
    "parse_audiobook_script",
    "parse_frontmatter",
    "split_into_sentences",
    "wav_to_base64_data_url",
]


def __getattr__(name: str):
    if name in ("WorkerEngine", "WorkerPool"):
        from .pool import WorkerEngine, WorkerPool
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
