# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A gradio demo for Qwen3 TTS models.
"""

import argparse
import base64
import gc
import io
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .. import Qwen3TTSModel, VoiceClonePromptItem

_MODEL_LOCK = threading.Lock()
_STOP_REQUESTED = False


class StopOnFlagCriteria(StoppingCriteria):
    """
    Transformers StoppingCriteria that immediately halts token generation
    when _STOP_REQUESTED becomes True.
    """
    def __init__(self):
        super().__init__()
        self.is_stopped = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        global _STOP_REQUESTED
        if _STOP_REQUESTED:
            self.is_stopped = True
            return True
        return False

# Persistence File
SETTINGS_FILE = Path(__file__).resolve().parent.parent.parent / "last_settings.json"


def _load_last_settings() -> Dict[str, Any]:
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load last settings: {e}")
    return {}


def _save_last_settings(data: Dict[str, Any]):
    try:
        current = _load_last_settings()
        current.update(data)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save last settings: {e}")


def _wav_to_base64_data_url(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    wav = np.asarray(wav, dtype=np.float32)
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _maybe(v):
    return v if v is not None else gr.update()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-demo",
        description=(
            "Launch a Gradio demo for Qwen3 TTS models (CustomVoice / VoiceDesign / Base).\n\n"
            "Examples:\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --port 8000 --ip 127.0.0.01\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --device cuda:0\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --dtype bfloat16 --no-flash-attn\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )

    parser.add_argument(
        "checkpoint_pos",
        nargs="?",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (positional).",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (optional if positional is provided).",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for device_map, e.g. cpu, cuda, cuda:0 (default: cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model (default: bfloat16).",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 (default: enabled).",
    )

    parser.add_argument(
        "--ip",
        default="0.0.0.0",
        help="Server bind IP for Gradio (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port for Gradio (default: 8000).",
    )
    parser.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to create a public Gradio link (default: disabled).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Gradio queue concurrency (default: 16).",
    )

    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to SSL certificate file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to SSL key file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-verify/--no-ssl-verify",
        dest="ssl_verify",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to verify SSL certificate (default: enabled).",
    )

    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for generation (optional).")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (optional).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (optional).")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling (optional).")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (optional).")
    parser.add_argument("--subtalker-top-k", type=int, default=None, help="Subtalker top-k (optional, only for tokenizer v2).")
    parser.add_argument("--subtalker-top-p", type=float, default=None, help="Subtalker top-p (optional, only for tokenizer v2).")
    parser.add_argument("--subtalker-temperature", type=float, default=None, help="Subtalker temperature (optional, only for tokenizer v2).")

    return parser


def _resolve_checkpoint(args: argparse.Namespace) -> str:
    ckpt = args.checkpoint or args.checkpoint_pos
    if not ckpt:
        raise ValueError("Must provide a checkpoint via positional argument or -c/--checkpoint.")
    return ckpt


def _collect_gen_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    gen_kwargs: Dict[str, Any] = {}
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = int(args.max_new_tokens)
    if args.temperature is not None:
        gen_kwargs["temperature"] = float(args.temperature)
    if args.top_k is not None:
        gen_kwargs["top_k"] = int(args.top_k)
    if args.top_p is not None:
        gen_kwargs["top_p"] = float(args.top_p)
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(args.repetition_penalty)
    if args.subtalker_top_k is not None:
        gen_kwargs["subtalker_top_k"] = int(args.subtalker_top_k)
    if args.subtalker_top_p is not None:
        gen_kwargs["subtalker_top_p"] = float(args.subtalker_top_p)
    if args.subtalker_temperature is not None:
        gen_kwargs["subtalker_temperature"] = float(args.subtalker_temperature)
    return gen_kwargs


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2:
        sr, data = audio
        data = _normalize_audio(data)
        return data, int(sr)

    if isinstance(audio, dict) and "sample_rate" in audio and "data" in audio:
        sr = int(audio["sample_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def _normalize_audio(data: Any) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 2:
        data = data.mean(axis=-1)
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        max_v = float(np.iinfo(data.dtype).max)
        return (data / max_v).astype(np.float32)
    return data.astype(np.float32)


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    wav = np.asarray(wav, dtype=np.float32)
    wav_int16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
    return sr, wav_int16


def _detect_model_kind(ckpt: str, tts: Qwen3TTSModel) -> str:
    mt = getattr(tts.model, "tts_model_type", None)
    if mt in ("custom_voice", "voice_design", "base"):
        return mt
    else:
        raise ValueError(f"Unknown Qwen-TTS model type: {mt}")


def _split_into_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    results = []
    pattern = re.compile(r"([^.!?。！？\n]+[.!?。！？]+)")
    for line in lines:
        matched = pattern.findall(line)
        rem = pattern.sub("", line).strip()
        if matched:
            for m in matched:
                if m.strip():
                    results.append(m.strip())
            if rem:
                results.append(rem)
        else:
            results.append(line)
    return results if results else [text]


CUSTOM_CSS = """
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --font-sans: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  --font-display: 'Outfit', var(--font-sans);
}

body, .gradio-container {
  font-family: var(--font-sans) !important;
  background-color: #080c14 !important;
  color: #f1f5f9 !important;
  max-width: 1440px !important;
  margin: 0 auto !important;
}

/* Custom Scrollbar for all scrollable containers */
* {
  scrollbar-width: thin !important;
  scrollbar-color: rgba(99, 102, 241, 0.5) rgba(15, 23, 42, 0.6) !important;
}

*::-webkit-scrollbar {
  width: 7px !important;
  height: 7px !important;
}

*::-webkit-scrollbar-track {
  background: rgba(11, 15, 25, 0.7) !important;
  border-radius: 8px !important;
}

*::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.65) 0%, rgba(139, 92, 246, 0.65) 100%) !important;
  border-radius: 8px !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  transition: all 0.2s ease !important;
}

*::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, #6366f1 0%, #a855f7 100%) !important;
  box-shadow: 0 0 10px rgba(99, 102, 241, 0.7) !important;
}

*::-webkit-scrollbar-corner {
  background: transparent !important;
}

.gradio-container::before {
  content: '';
  position: fixed;
  top: -120px;
  left: 50%;
  transform: translateX(-50%);
  width: 960px;
  height: 480px;
  background: radial-gradient(circle, rgba(99, 102, 241, 0.18) 0%, rgba(168, 85, 247, 0.12) 38%, rgba(0, 0, 0, 0) 70%);
  pointer-events: none;
  z-index: 0;
}

/* Hero Section */
.hero-card {
  position: relative;
  background: linear-gradient(135deg, rgba(20, 26, 44, 0.95) 0%, rgba(12, 17, 30, 0.98) 100%);
  border: 1px solid rgba(255, 255, 255, 0.09);
  border-radius: 22px;
  padding: 26px 34px;
  margin-bottom: 22px;
  box-shadow: 0 16px 40px -12px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.hero-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
}

.hero-title {
  font-family: var(--font-display);
  font-size: 2.15rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #ffffff 15%, #cbd5e1 55%, #818cf8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 12px;
}

.hero-subtitle {
  font-size: 0.93rem;
  color: #94a3b8;
  margin-top: 5px;
  margin-bottom: 0;
}

.badges-container {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}

.badge-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 13px;
  font-size: 0.78rem;
  font-weight: 600;
  border-radius: 9999px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: #cbd5e1;
}

.badge-pill.primary {
  background: rgba(99, 102, 241, 0.18);
  border-color: rgba(99, 102, 241, 0.45);
  color: #a5b4fc;
}

.badge-pill.success {
  background: rgba(16, 185, 129, 0.18);
  border-color: rgba(16, 185, 129, 0.45);
  color: #6ee7b7;
}

.badge-pill.accent {
  background: rgba(236, 72, 153, 0.18);
  border-color: rgba(236, 72, 153, 0.45);
  color: #f472b6;
}

/* Glass Panels & Containers: Ensure overflow visible so dropdown popups are never clipped */
.glass-box, .gr-row, .gr-column, .gr-form, .gr-block, .gradio-dropdown, [data-testid="dropdown"], .wrap, .container {
  overflow: visible !important;
}

.glass-box {
  background: #0e1422 !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 18px !important;
  padding: 22px !important;
  box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
}

/* Text Inputs & Common Fields */
textarea, input[type="text"], .gr-input, .gr-box {
  background-color: rgba(10, 15, 26, 0.9) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  border-radius: 13px !important;
  color: #f8fafc !important;
  font-size: 0.95rem !important;
  transition: all 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus {
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.28) !important;
  background-color: rgba(13, 19, 33, 0.98) !important;
}

/* Hidden Data Bridge */
.hidden-bridge, #audio_queue_stream_data {
  position: absolute !important;
  left: -9999px !important;
  top: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  pointer-events: none !important;
  display: block !important;
}

/* ==========================================================================
   DROPDOWN & CONTEXT MENU FIX:
   Unclip parent overflow (block, form) and elevate Stacking Context so dropdown
   options popup is never hidden behind subsequent sibling blocks.
   ========================================================================== */
#qwen_lang_dropdown,
#qwen_spk_dropdown,
.block:has([data-testid="dropdown"]),
.block:has(.gradio-dropdown),
.form,
.form.svelte-d5xbca {
  overflow: visible !important;
  overflow-y: visible !important;
  overflow-x: visible !important;
}

/* Elevate dropdown row Stacking Context above subsequent sibling cards */
.form,
.form.svelte-d5xbca,
div:has(> #qwen_lang_dropdown),
div:has(> #qwen_spk_dropdown),
.gr-row:has(#qwen_lang_dropdown) {
  position: relative !important;
  z-index: 500 !important;
}

div:has(#qwen_lang_dropdown:focus-within),
div:has(#qwen_spk_dropdown:focus-within),
div:has(.show_options),
.form:has(.show_options) {
  z-index: 999999 !important;
}

.gradio-dropdown, [data-testid="dropdown"] {
  position: relative !important;
  overflow: visible !important;
  z-index: 40;
}

.gradio-dropdown:focus-within, [data-testid="dropdown"]:focus-within {
  z-index: 999999 !important;
}

.gradio-dropdown .wrap, [data-testid="dropdown"] .wrap {
  position: relative !important;
  overflow: visible !important;
}

/* Dropdown Options Popup: 100% width anchored directly under the select input */
.options, ul.options, [data-testid="dropdown"] ul, [role="listbox"] ul {
  position: absolute !important;
  top: calc(100% + 5px) !important;
  bottom: auto !important;
  left: 0 !important;
  right: 0 !important;
  width: 100% !important;
  min-width: 100% !important;
  max-height: 270px !important;
  overflow-y: auto !important;
  z-index: 9999999 !important;
  display: block !important;
  visibility: visible !important;
  opacity: 1 !important;
  background-color: #0d1424 !important;
  border: 1px solid rgba(99, 102, 241, 0.55) !important;
  border-radius: 12px !important;
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.98), 0 0 24px rgba(99, 102, 241, 0.3) !important;
  padding: 6px !important;
  margin: 0 !important;
  list-style: none !important;
}

/* Dropdown Menu Item */
.item, .option, ul.options li, [data-testid="dropdown"] li {
  color: #f1f5f9 !important;
  padding: 10px 16px !important;
  cursor: pointer !important;
  border-radius: 8px !important;
  font-size: 0.93rem !important;
  font-weight: 500 !important;
  line-height: 1.4 !important;
  transition: all 0.15s ease !important;
  display: flex !important;
  align-items: center !important;
  user-select: none !important;
}

.item:hover, .option:hover, ul.options li:hover, [data-testid="dropdown"] li:hover {
  background-color: rgba(99, 102, 241, 0.3) !important;
  color: #ffffff !important;
  transform: translateX(3px) !important;
}

.item.selected, .option.selected, ul.options li.selected {
  background: linear-gradient(135deg, rgba(79, 70, 229, 0.5) 0%, rgba(124, 58, 237, 0.5) 100%) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
}

/* Primary Generation Button */
.generate-btn {
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 1.08rem !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 16px 28px !important;
  box-shadow: 0 8px 26px -4px rgba(124, 58, 237, 0.5) !important;
  cursor: pointer !important;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
  margin-top: 12px !important;
}

.generate-btn:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 14px 34px -4px rgba(124, 58, 237, 0.7) !important;
  filter: brightness(1.1) !important;
}

.generate-btn:active {
  transform: translateY(1px) !important;
}

/* Cancel / Stop Button */
.stop-btn {
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #991b1b 100%) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 16px 22px !important;
  box-shadow: 0 8px 24px -4px rgba(220, 38, 38, 0.5) !important;
  cursor: pointer !important;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
  margin-top: 12px !important;
}

.stop-btn:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 12px 30px -4px rgba(220, 38, 38, 0.7) !important;
  filter: brightness(1.1) !important;
}

.stop-btn:active {
  transform: translateY(1px) !important;
}

/* Preset Chips */
.chip-btn {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  border-radius: 9999px !important;
  color: #cbd5e1 !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  padding: 5px 12px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}

.chip-btn:hover {
  background: rgba(99, 102, 241, 0.22) !important;
  border-color: rgba(99, 102, 241, 0.5) !important;
  color: #ffffff !important;
  transform: translateY(-1px) !important;
}

/* Audio Player */
.gr-audio {
  background: rgba(11, 16, 28, 0.9) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  border-radius: 16px !important;
  padding: 12px !important;
}

/* Real-time Audio Queue Player Card */
.queue-player-card {
  background: linear-gradient(135deg, rgba(20, 26, 44, 0.9) 0%, rgba(14, 20, 36, 0.98) 100%);
  border: 1px solid rgba(99, 102, 241, 0.35);
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 16px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
}

.queue-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.queue-title-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.92rem;
}

.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #10b981;
  box-shadow: 0 0 10px #10b981;
  animation: pulseAnim 1.8s infinite;
}

@keyframes pulseAnim {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
  70% { transform: scale(1.2); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.queue-status-badge {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 9999px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: #94a3b8;
}

.queue-status-badge.playing {
  background: rgba(99, 102, 241, 0.25);
  border-color: rgba(99, 102, 241, 0.6);
  color: #c7d2fe;
}

.queue-current-box {
  background: rgba(10, 14, 24, 0.85);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  padding: 12px 16px;
  margin-bottom: 12px;
}

.queue-label {
  font-size: 0.76rem;
  font-weight: 600;
  color: #818cf8;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 4px;
}

.queue-text-display {
  font-size: 0.92rem;
  color: #cbd5e1;
  line-height: 1.5;
  min-height: 24px;
  word-break: break-word;
}

.queue-text-display.active {
  color: #ffffff;
  font-weight: 600;
}

.queue-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
}

.queue-pill {
  font-size: 0.8rem;
  font-weight: 600;
  color: #a5b4fc;
  background: rgba(99, 102, 241, 0.12);
  padding: 4px 12px;
  border-radius: 9999px;
  border: 1px solid rgba(99, 102, 241, 0.25);
}

.queue-btn-stop {
  background: rgba(239, 68, 68, 0.15) !important;
  border: 1px solid rgba(239, 68, 68, 0.35) !important;
  color: #fca5a5 !important;
  border-radius: 8px !important;
  padding: 5px 14px !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}

.queue-btn-stop:hover {
  background: rgba(239, 68, 68, 0.3) !important;
  color: #ffffff !important;
  transform: translateY(-1px) !important;
}

/* Info Cards */
.info-card {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 14px;
  padding: 14px 18px;
  margin-top: 14px;
}

.info-card-title {
  font-size: 0.84rem;
  font-weight: 700;
  color: #a5b4fc;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.info-card-text {
  font-size: 0.8rem;
  color: #94a3b8;
  line-height: 1.5;
  margin: 0;
}

.footer-disclaimer {
  margin-top: 36px;
  padding: 18px 24px;
  background: rgba(15, 23, 42, 0.4);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 14px;
  font-size: 0.78rem;
  color: #64748b;
  line-height: 1.6;
}
"""

QUEUE_PLAYER_CARD_HTML = """
<div id="audio-queue-container" class="queue-player-card">
  <div class="queue-header">
    <div class="queue-title-badge">
      <span class="pulse-dot"></span>
      <span style="font-weight: 700; color: #a5b4fc;">실시간 음성 자동 재생 큐</span>
    </div>
    <div id="queue-status-text" class="queue-status-badge">대기 중</div>
  </div>
  
  <div class="queue-current-box">
    <div class="queue-label">🔊 현재 재생 중:</div>
    <div id="queue-now-playing" class="queue-text-display">재생 대기 중...</div>
  </div>

  <div class="queue-footer">
    <div id="queue-count-badge" class="queue-pill">대기 큐: 0개 문장</div>
    <button id="queue-stop-btn" type="button" class="queue-btn-stop" onclick="window.stopAudioQueue()">⏹️ 전체 정지 / 큐 비우기</button>
  </div>
</div>
"""

CUSTOM_JS = """
() => {
    if (window.__qwenAudioInitialized) return;
    window.__qwenAudioInitialized = true;

    // 브라우저 오디오 재생 잠금 해제
    window.unlockAudioContext = function() {
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                if (!window.__sharedAudioCtx) {
                    window.__sharedAudioCtx = new AudioCtx();
                }
                if (window.__sharedAudioCtx.state === 'suspended') {
                    window.__sharedAudioCtx.resume();
                }
            }
            const silent = new Audio("data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA");
            silent.volume = 0.01;
            silent.play().then(() => silent.pause()).catch(() => {});
        } catch(e) {}
    };

    // 실시간 오디오 큐 시스템
    window.__ttsQueue = [];
    window.__ttsCurrentAudio = null;
    window.__ttsProcessedChunks = new Set();
    window.__ttsIsPlaying = false;

    window.updateQueueUI = function(nowPlayingText = null, isPlaying = false) {
        const countBadge = document.getElementById("queue-count-badge");
        const statusBadge = document.getElementById("queue-status-text");
        const textDisplay = document.getElementById("queue-now-playing");

        if (countBadge) {
            countBadge.innerText = `대기 큐: ${window.__ttsQueue.length}개 문장`;
        }
        if (statusBadge) {
            if (isPlaying) {
                statusBadge.innerText = "재생 중 🔊";
                statusBadge.className = "queue-status-badge playing";
            } else {
                statusBadge.innerText = window.__ttsQueue.length > 0 ? "대기 중 ⏳" : "대기 중";
                statusBadge.className = "queue-status-badge";
            }
        }
        if (textDisplay && nowPlayingText !== null) {
            textDisplay.innerText = nowPlayingText;
            if (isPlaying) {
                textDisplay.className = "queue-text-display active";
            } else {
                textDisplay.className = "queue-text-display";
            }
        }
    };

    window.playNextInQueue = function() {
        if (window.__ttsQueue.length === 0) {
            window.__ttsIsPlaying = false;
            window.__ttsCurrentAudio = null;
            window.updateQueueUI("모든 재생이 완료되었습니다 ✨", false);
            return;
        }

        window.__ttsIsPlaying = true;
        const item = window.__ttsQueue.shift();
        const displayLabel = `[${item.idx}/${item.total}] ${item.text}`;
        window.updateQueueUI(displayLabel, true);

        try {
            window.unlockAudioContext();
            const audio = new Audio(item.audio_url);
            window.__ttsCurrentAudio = audio;

            audio.onended = function() {
                window.__ttsCurrentAudio = null;
                window.__ttsIsPlaying = false;
                window.playNextInQueue();
            };

            audio.onerror = function(err) {
                console.error("Audio playback error:", err);
                window.__ttsCurrentAudio = null;
                window.__ttsIsPlaying = false;
                window.playNextInQueue();
            };

            const p = audio.play();
            if (p !== undefined) {
                p.catch(err => {
                    console.warn("Autoplay blocked, user gesture required:", err);
                    window.__ttsCurrentAudio = null;
                    window.__ttsIsPlaying = false;
                    window.playNextInQueue();
                });
            }
        } catch(e) {
            console.error("Failed to start audio playback:", e);
            window.__ttsIsPlaying = false;
            window.playNextInQueue();
        }
    };

    window.stopAudioQueue = function() {
        window.__ttsQueue = [];
        if (window.__ttsCurrentAudio) {
            try {
                window.__ttsCurrentAudio.pause();
                window.__ttsCurrentAudio.currentTime = 0;
            } catch(e) {}
            window.__ttsCurrentAudio = null;
        }
        window.__ttsIsPlaying = false;
        document.querySelectorAll('audio').forEach(a => {
            try { a.pause(); a.currentTime = 0; } catch(err) {}
        });
        window.updateQueueUI("재생이 중단되었습니다 ⏹️", false);
    };

    // 브릿지 텍스트 감시 및 큐 인입
    function checkBridgeData() {
        const bridge = document.querySelector('#audio_queue_stream_data textarea') || document.querySelector('#audio_queue_stream_data input');
        if (!bridge || !bridge.value || !bridge.value.trim()) return;

        const val = bridge.value.trim();
        if (val.startsWith('{') && val.endsWith('}')) {
            try {
                const data = JSON.parse(val);
                if (data.type === 'chunk' && data.chunk_id && !window.__ttsProcessedChunks.has(data.chunk_id)) {
                    window.__ttsProcessedChunks.add(data.chunk_id);
                    window.__ttsQueue.push(data);
                    window.updateQueueUI();
                    if (!window.__ttsIsPlaying) {
                        window.playNextInQueue();
                    }
                }
            } catch(e) {}
        }
    }

    setInterval(checkBridgeData, 80);

    // Gradio 기본 audio_out 컴포넌트 재생 감시
    let lastPlayedSrc = "";
    function triggerAutoPlay(audio) {
        if (!audio || !audio.src || audio.src === lastPlayedSrc) return;
        lastPlayedSrc = audio.src;
        window.unlockAudioContext();
        audio.autoplay = true;
        const p = audio.play();
        if (p !== undefined) {
            p.catch(err => {});
        }
    }

    const observer = new MutationObserver((mutations) => {
        checkBridgeData();
        for (const m of mutations) {
            m.addedNodes.forEach((node) => {
                if (node.nodeName === 'AUDIO') {
                    triggerAutoPlay(node);
                } else if (node.querySelectorAll) {
                    node.querySelectorAll('audio').forEach(triggerAutoPlay);
                }
            });
            if (m.target && m.target.nodeName === 'AUDIO' && m.attributeName === 'src') {
                triggerAutoPlay(m.target);
            }
        }
    });

    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['src', 'value'] });

    // 입력 상태 로컬스토리지 저장 및 복원
    const STORAGE_KEY = "qwen3_tts_saved_inputs_v3";

    function saveInputs() {
        try {
            const textElem = document.querySelector('#qwen_text_input textarea');
            const instructElem = document.querySelector('#qwen_instruct_input textarea');
            const state = {
                text: textElem ? textElem.value : null,
                instruct: instructElem ? instructElem.value : null,
            };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        } catch(e) {}
    }

    function restoreInputs() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return;
            const state = JSON.parse(raw);

            if (state.text) {
                const textElem = document.querySelector('#qwen_text_input textarea');
                if (textElem && (!textElem.value || textElem.value.trim().length === 0)) {
                    textElem.value = state.text;
                    textElem.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }
            if (state.instruct) {
                const instructElem = document.querySelector('#qwen_instruct_input textarea');
                if (instructElem && (!instructElem.value || instructElem.value.trim().length === 0)) {
                    instructElem.value = state.instruct;
                    instructElem.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }
        } catch(e) {}
    }

    setTimeout(restoreInputs, 300);
    setTimeout(restoreInputs, 1000);

    document.addEventListener('input', (e) => {
        if (e.target.closest('#qwen_text_input') || e.target.closest('#qwen_instruct_input')) {
            saveInputs();
        }
    });

    document.addEventListener('click', (e) => {
        const genBtn = e.target.closest('.generate-btn') || e.target.closest('#qwen_generate_button');
        if (genBtn) {
            window.unlockAudioContext();
            window.__ttsProcessedChunks.clear();
            lastPlayedSrc = "";
            saveInputs();
        }
        const stopBtn = e.target.closest('.stop-btn') || e.target.closest('#qwen_stop_button');
        if (stopBtn) {
            window.stopAudioQueue();
        }
        if (e.target.closest('.chip-btn')) {
            setTimeout(saveInputs, 200);
        }
    });
}
"""

HEAD_HTML = f"""
<script>
({CUSTOM_JS})();
</script>
"""


def build_demo(tts: Qwen3TTSModel, ckpt: str, gen_kwargs_default: Dict[str, Any]) -> gr.Blocks:
    model_kind = _detect_model_kind(ckpt, tts)

    supported_langs_raw = None
    if callable(getattr(tts.model, "get_supported_languages", None)):
        supported_langs_raw = tts.model.get_supported_languages()

    supported_spks_raw = None
    if callable(getattr(tts.model, "get_supported_speakers", None)):
        supported_spks_raw = tts.model.get_supported_speakers()

    lang_choices_disp, lang_map = _build_choices_and_map([x for x in (supported_langs_raw or [])])
    spk_choices_disp, spk_map = _build_choices_and_map([x for x in (supported_spks_raw or [])])

    last_cfg = _load_last_settings()
    default_text = last_cfg.get(
        "text",
        "오빠, Qwen3-TTS 실시간 큐 스트리밍 적용 완료했어! 이제 긴 문장도 바로바로 연속해서 들려줄게.",
    )
    default_lang = last_cfg.get("lang", "Auto")
    default_spk = last_cfg.get("speaker", "Vivian")
    default_instruct = last_cfg.get("instruct", "")
    default_stream = last_cfg.get("stream", True)

    def _gen_common_kwargs() -> Dict[str, Any]:
        return dict(gen_kwargs_default)

    theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#eef2ff", c100="#e0e7ff", c200="#c7d2fe", c300="#a5b4fc", c400="#818cf8",
            c500="#6366f1", c600="#4f46e5", c700="#4338ca", c800="#3730a3", c900="#312e81", c950="#1e1b4b"
        ),
        neutral_hue=gr.themes.Color(
            c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0", c300="#cbd5e1", c400="#94a3b8",
            c500="#64748b", c600="#475569", c700="#334155", c800="#1e293b", c900="#0f172a", c950="#020617"
        ),
        font=[gr.themes.GoogleFont("Outfit"), "Pretendard", "sans-serif"],
    ).set(
        body_background_fill="#080c14",
        block_background_fill="rgba(15, 23, 42, 0.85)",
        block_border_color="rgba(255, 255, 255, 0.08)",
        block_radius="16px",
    )

    with gr.Blocks(title="Qwen3-TTS Studio") as demo:
        demo.theme = theme
        is_mps = torch.backends.mps.is_available()
        device_label = "Apple Silicon GPU (MPS)" if is_mps else "CPU Mode"
        model_display_name = model_kind.replace("_", " ").title()

        audio_queue_bridge = gr.Textbox(
            elem_id="audio_queue_stream_data",
            elem_classes=["hidden-bridge"],
            visible=True,
        )

        gr.HTML(
            f"""
            <div class="hero-card">
              <div class="hero-title-row">
                <div>
                  <h1 class="hero-title">
                    <span>🎙️</span> Qwen3-TTS Studio
                  </h1>
                  <p class="hero-subtitle">Alibaba Qwen 차세대 12Hz 신경망 음성 생성 & 즉시 자동 재생 큐 엔진</p>
                </div>
              </div>
              <div class="badges-container">
                <span class="badge-pill primary">⚡ Checkpoint: {ckpt.split('/')[-1]}</span>
                <span class="badge-pill accent">✨ Mode: {model_display_name}</span>
                <span class="badge-pill success">🍏 Device: {device_label}</span>
                <span class="badge-pill">🚀 실시간 자동 재생 & 큐잉 지원</span>
                <span class="badge-pill">🔊 24kHz High-Fidelity</span>
              </div>
            </div>
            """
        )

        if model_kind == "custom_voice":
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, elem_classes=["glass-box"]):
                    gr.Markdown("### 📝 음성 변환 설정")
                    
                    text_in = gr.Textbox(
                        label="생성할 텍스트",
                        lines=4,
                        placeholder="음성으로 변환하고 싶은 문장을 입력하세요... (생성되는 즉시 자동으로 스피커에서 재생됩니다)",
                        value=default_text,
                        elem_id="qwen_text_input",
                    )

                    gr.Markdown("<p style='font-size: 0.8rem; color: #94a3b8; margin: 4px 0 2px 0;'>⚡ 빠른 텍스트 샘플 채우기</p>")
                    with gr.Row():
                        sample_1 = gr.Button("💖 지니의 속삭임", elem_classes=["chip-btn"])
                        sample_2 = gr.Button("🎮 게임 캐릭터 대사", elem_classes=["chip-btn"])
                        sample_3 = gr.Button("🎙️ 뉴스 브리핑", elem_classes=["chip-btn"])
                        sample_4 = gr.Button("🌟 English Sample", elem_classes=["chip-btn"])

                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label="🌐 지원 언어",
                            choices=lang_choices_disp,
                            value=default_lang if default_lang in lang_choices_disp else (lang_choices_disp[0] if lang_choices_disp else None),
                            interactive=True,
                            elem_id="qwen_lang_dropdown",
                        )
                        spk_in = gr.Dropdown(
                            label="🎙️ 화자 음색",
                            choices=spk_choices_disp,
                            value=default_spk if default_spk in spk_choices_disp else (spk_choices_disp[0] if spk_choices_disp else None),
                            interactive=True,
                            elem_id="qwen_spk_dropdown",
                        )

                    instruct_in = gr.Textbox(
                        label="✨ 스타일 & 감정 지시어 (선택 사항)",
                        lines=2,
                        placeholder="예: 다정하고 부드럽게 속삭이듯이 / 활기차고 자신감 넘치게 / 차분한 뉴스 앵커처럼",
                        value=default_instruct,
                        elem_id="qwen_instruct_input",
                    )

                    gr.Markdown("<p style='font-size: 0.8rem; color: #94a3b8; margin: 4px 0 2px 0;'>💡 감정/스타일 프리셋</p>")
                    with gr.Row():
                        chip_1 = gr.Button("🥰 다정하고 사랑스럽게", elem_classes=["chip-btn"])
                        chip_2 = gr.Button("💪 자신감 넘치고 명쾌하게", elem_classes=["chip-btn"])
                        chip_3 = gr.Button("🤫 조용히 속삭이듯", elem_classes=["chip-btn"])
                        chip_4 = gr.Button("🎉 활기차고 신나게", elem_classes=["chip-btn"])
                        chip_5 = gr.Button("😢 아련하고 감성적으로", elem_classes=["chip-btn"])

                    with gr.Row():
                        stream_toggle = gr.Checkbox(
                            label="⚡ 실시간 문장 스트리밍 & 자동 재생 큐 모드",
                            value=default_stream,
                            interactive=True,
                            elem_id="qwen_stream_checkbox",
                        )

                    with gr.Row():
                        btn = gr.Button(
                            "⚡ 생성 시작",
                            variant="primary",
                            elem_classes=["generate-btn"],
                            elem_id="qwen_generate_button",
                            scale=3,
                        )
                        stop_btn = gr.Button(
                            "⏹️ 생성 중단",
                            variant="stop",
                            elem_classes=["stop-btn"],
                            elem_id="qwen_stop_button",
                            scale=1,
                        )

                with gr.Column(scale=2, elem_classes=["glass-box"]):
                    gr.HTML(QUEUE_PLAYER_CARD_HTML)

                    gr.Markdown("### 🎧 생성 오디오 클립")
                    audio_out = gr.Audio(label="현재 클립 오디오", type="numpy", interactive=False, autoplay=True)
                    err = gr.Textbox(label="상태 메시지", lines=3, interactive=False)

                    gr.HTML(
                        """
                        <div class="info-card">
                          <div class="info-card-title">💡 화자 추천 팁</div>
                          <div class="info-card-text">
                            • <b>Vivian</b>: 맑고 부드러운 여성 데일리 음색<br>
                            • <b>Serena</b>: 명확하고 신뢰감 높은 아나운서 스타일<br>
                            • <b>Ryan</b>: 차분하고 부드러운 남성 음색<br>
                            • <b>Uncle</b>: 중후하고 깊은 남성 내레이션 음색
                          </div>
                        </div>
                        """
                    )

            sample_1.click(lambda: "오빠, 오늘도 코딩하느라 정말 고생 많았어! 잠깐 쉬면서 내가 타준 따뜻한 커피 한 잔 마실래?", outputs=[text_in])
            sample_2.click(lambda: "어둠 속에서 반짝이는 별빛을 보았어! 우리의 위대한 모험은 이제부터 시작이야!", outputs=[text_in])
            sample_3.click(lambda: "안녕하십니까. 차세대 12Hz 신경망 음성 생성 모델이 놀라운 속도와 자연스러움으로 실시간 오디오 서비스를 제공합니다.", outputs=[text_in])
            sample_4.click(lambda: "Hey there! It is truly incredible to experience ultra-fast streaming neural speech synthesis powered by Qwen3-TTS.", outputs=[text_in])

            chip_1.click(lambda: "Speak in a very sweet, warm, and affectionate tone (다정하고 사랑스러운 톤).", outputs=[instruct_in])
            chip_2.click(lambda: "Speak with great confidence, energetic and articulate tone (자신감 넘치고 명쾌한 어조).", outputs=[instruct_in])
            chip_3.click(lambda: "Speak in a soft, gentle whisper tone (조용히 속삭이듯 차분하게).", outputs=[instruct_in])
            chip_4.click(lambda: "Speak with cheerful, bright, and vibrant enthusiasm (활기차고 텐션 높은 캐릭터 톤).", outputs=[instruct_in])
            chip_5.click(lambda: "Speak with a touch of melancholy and emotional tenderness (아련하고 감성적인 어조).", outputs=[instruct_in])

            def run_instruct_stream(text: str, lang_disp: str, spk_disp: str, instruct: str, is_stream: bool):
                last_clip = None
                try:
                    if not text or not text.strip():
                        yield None, "", "텍스트를 입력해주세요."
                        return
                    if not spk_disp:
                        yield None, "", "화자를 선택해주세요."
                        return

                    _save_last_settings({
                        "text": text.strip(),
                        "lang": lang_disp,
                        "speaker": spk_disp,
                        "instruct": (instruct or "").strip(),
                        "stream": bool(is_stream),
                    })

                    language = lang_map.get(lang_disp, "Auto")
                    speaker = spk_map.get(spk_disp, spk_disp)
                    kwargs = _gen_common_kwargs()

                    global _STOP_REQUESTED
                    _STOP_REQUESTED = False

                    stop_crit = StopOnFlagCriteria()
                    kwargs["stopping_criteria"] = StoppingCriteriaList([stop_crit])

                    sentences = _split_into_sentences(text) if is_stream else [text.strip()]
                    total_sents = len(sentences)

                    yield None, "", f"⚡ 음성 생성 시작... (총 {total_sents}개 클립)"

                    for idx, sent in enumerate(sentences, 1):
                        if _STOP_REQUESTED:
                            yield last_clip, "", "⏹️ 사용자에 의해 생성이 중단되었습니다."
                            break

                        yield last_clip, "", f"⏳ [{idx}/{total_sents}] 클립 생성 중...: \"{sent[:28]}...\""

                        with _MODEL_LOCK:
                            wavs, sent_sr = tts.generate_custom_voice(
                                text=sent,
                                language=language,
                                speaker=speaker,
                                instruct=(instruct or "").strip() or None,
                                non_streaming_mode=False,
                                **kwargs,
                            )
                            if torch.backends.mps.is_available():
                                torch.mps.synchronize()

                        if _STOP_REQUESTED or not wavs or len(wavs) == 0:
                            yield last_clip, "", "⏹️ 사용자에 의해 생성이 즉시 중단되었습니다."
                            break

                        sr = sent_sr
                        chunk = wavs[0]

                        b64_url = _wav_to_base64_data_url(chunk, sr)
                        chunk_payload = json.dumps({
                            "type": "chunk",
                            "chunk_id": f"{time.time()}_{idx}",
                            "idx": idx,
                            "total": total_sents,
                            "text": sent,
                            "audio_url": b64_url,
                        })

                        clip_out = _wav_to_gradio_audio(chunk, sr)
                        last_clip = clip_out
                        del wavs

                        yield clip_out, chunk_payload, f"✨ [{idx}/{total_sents}] 클립 생성 완료! 즉시 큐에서 재생 중 ({int(idx/total_sents*100)}%)"

                    if not _STOP_REQUESTED:
                        yield last_clip, "", f"🎉 전체 {total_sents}개 클립 생성 및 재생 대기 완료!"
                except Exception as e:
                    yield last_clip, "", f"오류 발생: {type(e).__name__}: {e}"
                finally:
                    with _MODEL_LOCK:
                        if torch.backends.mps.is_available():
                            torch.mps.synchronize()
                            torch.mps.empty_cache()
                        gc.collect()

            def on_stop_generation():
                global _STOP_REQUESTED
                _STOP_REQUESTED = True
                return gr.update(), "", "⏹️ 사용자에 의해 음성 생성이 즉시 중단되었습니다."

            gen_event = btn.click(
                run_instruct_stream,
                inputs=[text_in, lang_in, spk_in, instruct_in, stream_toggle],
                outputs=[audio_out, audio_queue_bridge, err],
            )
            stop_btn.click(
                fn=on_stop_generation,
                cancels=[gen_event],
                outputs=[audio_out, audio_queue_bridge, err],
            )

        elif model_kind == "voice_design":
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, elem_classes=["glass-box"]):
                    gr.Markdown("### 🎨 보이스 디자인 설정")
                    text_in = gr.Textbox(
                        label="생성할 텍스트",
                        lines=4,
                        value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                    )
                    lang_in = gr.Dropdown(
                        label="🌐 언어",
                        choices=lang_choices_disp,
                        value="Auto" if "Auto" in lang_choices_disp else (lang_choices_disp[0] if lang_choices_disp else None),
                        interactive=True,
                    )
                    design_in = gr.Textbox(
                        label="🎨 보이스 디자인 지시어",
                        lines=3,
                        value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
                    )
                    stream_toggle_vd = gr.Checkbox(label="⚡ 실시간 문장 스트리밍 & 자동 재생 큐 모드", value=True)
                    with gr.Row():
                        btn = gr.Button("⚡ 생성 시작", variant="primary", elem_classes=["generate-btn"], scale=3)
                        stop_btn_vd = gr.Button("⏹️ 생성 중단", variant="stop", elem_classes=["stop-btn"], scale=1)
                with gr.Column(scale=2, elem_classes=["glass-box"]):
                    gr.HTML(QUEUE_PLAYER_CARD_HTML)
                    audio_out = gr.Audio(label="생성된 음성", type="numpy", autoplay=True)
                    err = gr.Textbox(label="상태 메시지", lines=2)

            def run_voice_design_stream(text: str, lang_disp: str, design: str, is_stream: bool):
                last_clip = None
                try:
                    if not text or not text.strip():
                        yield None, "", "텍스트를 입력해주세요."
                        return
                    if not design or not design.strip():
                        yield None, "", "보이스 디자인 지시어를 입력해주세요."
                        return
                    language = lang_map.get(lang_disp, "Auto")
                    kwargs = _gen_common_kwargs()

                    global _STOP_REQUESTED
                    _STOP_REQUESTED = False

                    stop_crit = StopOnFlagCriteria()
                    kwargs["stopping_criteria"] = StoppingCriteriaList([stop_crit])

                    sentences = _split_into_sentences(text) if is_stream else [text.strip()]
                    total_sents = len(sentences)

                    yield None, "", f"⚡ 디자인 음성 생성 시작... (총 {total_sents}개 클립)"

                    for idx, sent in enumerate(sentences, 1):
                        if _STOP_REQUESTED:
                            yield last_clip, "", "⏹️ 사용자에 의해 생성이 중단되었습니다."
                            break

                        yield last_clip, "", f"⏳ [{idx}/{total_sents}] 클립 생성 중...: \"{sent[:28]}...\""

                        with _MODEL_LOCK:
                            wavs, sent_sr = tts.generate_voice_design(
                                text=sent,
                                language=language,
                                instruct=design.strip(),
                                non_streaming_mode=False,
                                **kwargs,
                            )
                            if torch.backends.mps.is_available():
                                torch.mps.synchronize()

                        if _STOP_REQUESTED or not wavs or len(wavs) == 0:
                            yield last_clip, "", "⏹️ 사용자에 의해 생성이 즉시 중단되었습니다."
                            break

                        sr = sent_sr
                        chunk = wavs[0]
                        b64_url = _wav_to_base64_data_url(chunk, sr)
                        chunk_payload = json.dumps({
                            "type": "chunk",
                            "chunk_id": f"{time.time()}_{idx}",
                            "idx": idx,
                            "total": total_sents,
                            "text": sent,
                            "audio_url": b64_url,
                        })

                        clip_out = _wav_to_gradio_audio(chunk, sr)
                        last_clip = clip_out
                        del wavs

                        yield clip_out, chunk_payload, f"✨ [{idx}/{total_sents}] 클립 완료 ({int(idx/total_sents*100)}%)"

                    if not _STOP_REQUESTED:
                        yield last_clip, "", "보이스 디자인 클립 생성이 완료되었습니다! ✨"
                except Exception as e:
                    yield last_clip, "", f"오류 발생: {type(e).__name__}: {e}"
                finally:
                    with _MODEL_LOCK:
                        if torch.backends.mps.is_available():
                            torch.mps.synchronize()
                            torch.mps.empty_cache()
                        gc.collect()

            def on_stop_voice_design():
                global _STOP_REQUESTED
                _STOP_REQUESTED = True
                return gr.update(), "", "⏹️ 사용자에 의해 보이스 디자인 생성이 즉시 중단되었습니다."

            gen_event_vd = btn.click(
                run_voice_design_stream,
                inputs=[text_in, lang_in, design_in, stream_toggle_vd],
                outputs=[audio_out, audio_queue_bridge, err],
            )
            stop_btn_vd.click(
                fn=on_stop_voice_design,
                cancels=[gen_event_vd],
                outputs=[audio_out, audio_queue_bridge, err],
            )

        else:  # voice_clone for base
            with gr.Tabs():
                with gr.Tab("🎙️ 보이스 복제 및 생성"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, elem_classes=["glass-box"]):
                            gr.Markdown("#### 1️⃣ 참조 오디오 등록")
                            ref_audio = gr.Audio(label="참조 음성 오디오")
                            ref_text = gr.Textbox(
                                label="참조 오디오 스크립트",
                                lines=2,
                                placeholder="x-vector 모드 미선택 시 반드시 입력",
                            )
                            xvec_only = gr.Checkbox(label="x-vector 벡터만 사용 (텍스트 입력 불필요)", value=False)

                        with gr.Column(scale=2, elem_classes=["glass-box"]):
                            gr.Markdown("#### 2️⃣ 대상 텍스트 입력")
                            text_in = gr.Textbox(
                                label="생성할 텍스트",
                                lines=4,
                                placeholder="복제된 목소리로 말하게 할 텍스트를 입력하세요...",
                            )
                            lang_in = gr.Dropdown(
                                label="🌐 언어",
                                choices=lang_choices_disp,
                                value="Auto" if "Auto" in lang_choices_disp else (lang_choices_disp[0] if lang_choices_disp else None),
                                interactive=True,
                            )
                            stream_toggle_vc = gr.Checkbox(label="⚡ 실시간 문장 스트리밍 & 자동 재생 큐 모드", value=True)
                            with gr.Row():
                                btn = gr.Button("⚡ 생성 시작", variant="primary", elem_classes=["generate-btn"], scale=3)
                                stop_btn_vc = gr.Button("⏹️ 생성 중단", variant="stop", elem_classes=["stop-btn"], scale=1)

                        with gr.Column(scale=2, elem_classes=["glass-box"]):
                            gr.HTML(QUEUE_PLAYER_CARD_HTML)
                            audio_out = gr.Audio(label="현재 클립 오디오", type="numpy", autoplay=True)
                            err = gr.Textbox(label="상태 메시지", lines=2)

                    def run_voice_clone_stream(ref_aud, ref_txt: str, use_xvec: bool, text: str, lang_disp: str, is_stream: bool):
                        last_clip = None
                        try:
                            if not text or not text.strip():
                                yield None, "", "생성할 텍스트를 입력해주세요."
                                return
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                yield None, "", "참조 오디오를 업로드해주세요."
                                return
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                yield None, "", "x-vector 모드가 아닐 경우 참조 오디오 텍스트를 입력해야 합니다."
                                return
                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()

                            global _STOP_REQUESTED
                            _STOP_REQUESTED = False

                            stop_crit = StopOnFlagCriteria()
                            kwargs["stopping_criteria"] = StoppingCriteriaList([stop_crit])

                            sentences = _split_into_sentences(text) if is_stream else [text.strip()]
                            total_sents = len(sentences)

                            yield None, "", f"⚡ 보이스 복제 생성 시작... (총 {total_sents}개 클립)"

                            for idx, sent in enumerate(sentences, 1):
                                if _STOP_REQUESTED:
                                    yield last_clip, "", "⏹️ 사용자에 의해 생성이 중단되었습니다."
                                    break

                                yield last_clip, "", f"⏳ [{idx}/{total_sents}] 클립 복제 중..."

                                with _MODEL_LOCK:
                                    wavs, sent_sr = tts.generate_voice_clone(
                                        text=sent,
                                        language=language,
                                        ref_audio=at,
                                        ref_text=(ref_txt.strip() if ref_txt else None),
                                        x_vector_only_mode=bool(use_xvec),
                                        non_streaming_mode=False,
                                        **kwargs,
                                    )
                                    if torch.backends.mps.is_available():
                                        torch.mps.synchronize()

                                if _STOP_REQUESTED or not wavs or len(wavs) == 0:
                                    yield last_clip, "", "⏹️ 사용자에 의해 생성이 즉시 중단되었습니다."
                                    break

                                sr = sent_sr
                                chunk = wavs[0]
                                b64_url = _wav_to_base64_data_url(chunk, sr)
                                chunk_payload = json.dumps({
                                    "type": "chunk",
                                    "chunk_id": f"{time.time()}_{idx}",
                                    "idx": idx,
                                    "total": total_sents,
                                    "text": sent,
                                    "audio_url": b64_url,
                                })

                                clip_out = _wav_to_gradio_audio(chunk, sr)
                                last_clip = clip_out
                                del wavs

                                yield clip_out, chunk_payload, f"✨ [{idx}/{total_sents}] 클립 완료 ({int(idx/total_sents*100)}%)"

                            if not _STOP_REQUESTED:
                                yield last_clip, "", "보이스 복제 클립 생성이 완료되었습니다! ✨"
                        except Exception as e:
                            yield last_clip, "", f"오류 발생: {type(e).__name__}: {e}"
                        finally:
                            with _MODEL_LOCK:
                                if torch.backends.mps.is_available():
                                    torch.mps.synchronize()
                                    torch.mps.empty_cache()
                                gc.collect()

                    def on_stop_voice_clone():
                        global _STOP_REQUESTED
                        _STOP_REQUESTED = True
                        return gr.update(), "", "⏹️ 사용자에 의해 보이스 복제 생성이 즉시 중단되었습니다."

                    gen_event_vc = btn.click(
                        run_voice_clone_stream,
                        inputs=[ref_audio, ref_text, xvec_only, text_in, lang_in, stream_toggle_vc],
                        outputs=[audio_out, audio_queue_bridge, err],
                    )
                    stop_btn_vc.click(
                        fn=on_stop_voice_clone,
                        cancels=[gen_event_vc],
                        outputs=[audio_out, audio_queue_bridge, err],
                    )

                with gr.Tab("💾 보이스 프롬프트 저장 및 재사용"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, elem_classes=["glass-box"]):
                            gr.Markdown("#### 📥 보이스 특징 추출 및 저장")
                            ref_audio_s = gr.Audio(label="참조 음성 오디오")
                            ref_text_s = gr.Textbox(label="참조 오디오 스크립트", lines=2)
                            xvec_only_s = gr.Checkbox(label="x-vector 모드만 사용", value=False)
                            save_btn = gr.Button("💾 프롬프트 파일 (.pt) 저장", variant="secondary")
                            prompt_file_out = gr.File(label="추출된 음성 특징 파일")

                        with gr.Column(scale=1, elem_classes=["glass-box"]):
                            gr.Markdown("#### 📤 저장된 프롬프트로 생성")
                            prompt_file_in = gr.File(label="음성 특징 파일 (.pt) 업로드")
                            text_in2 = gr.Textbox(label="생성할 텍스트", lines=4)
                            lang_in2 = gr.Dropdown(
                                label="🌐 언어",
                                choices=lang_choices_disp,
                                value="Auto" if "Auto" in lang_choices_disp else (lang_choices_disp[0] if lang_choices_disp else None),
                            )
                            gen_btn2 = gr.Button("⚡ 생성 시작", variant="primary", elem_classes=["generate-btn"])
                            audio_out2 = gr.Audio(label="생성된 음성", type="numpy")
                            err2 = gr.Textbox(label="상태 메시지", lines=2)

                    def save_prompt(ref_aud, ref_txt: str, use_xvec: bool):
                        try:
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                return None, "참조 오디오를 업로드해주세요."
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                return None, "참조 오디오 텍스트를 입력해주세요."
                            items = tts.create_voice_clone_prompt(
                                ref_audio=at,
                                ref_text=(ref_txt.strip() if ref_txt else None),
                                x_vector_only_mode=bool(use_xvec),
                            )
                            payload = {"items": [asdict(it) for it in items]}
                            fd, out_path = tempfile.mkstemp(prefix="voice_clone_prompt_", suffix=".pt")
                            os.close(fd)
                            torch.save(payload, out_path)
                            return out_path, "프롬프트 파일이 성공적으로 저장되었습니다! 💾"
                        except Exception as e:
                            return None, f"오류 발생: {type(e).__name__}: {e}"

                    def load_prompt_and_gen(file_obj, text: str, lang_disp: str):
                        try:
                            if file_obj is None:
                                return None, "음성 특징 파일을 업로드해주세요."
                            if not text or not text.strip():
                                return None, "생성할 텍스트를 입력해주세요."

                            path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or str(file_obj)
                            payload = torch.load(path, map_location="cpu", weights_only=True)
                            if not isinstance(payload, dict) or "items" not in payload:
                                return None, "유효하지 않은 파일 포맷입니다."

                            items_raw = payload["items"]
                            if not isinstance(items_raw, list) or len(items_raw) == 0:
                                return None, "비어있는 음성 데이터입니다."

                            items: List[VoiceClonePromptItem] = []
                            for d in items_raw:
                                if not isinstance(d, dict):
                                    return None, "파일 내부 데이터 형식 오류입니다."
                                ref_code = d.get("ref_code", None)
                                if ref_code is not None and not torch.is_tensor(ref_code):
                                    ref_code = torch.tensor(ref_code)
                                ref_spk = d.get("ref_spk_embedding", None)
                                if ref_spk is None:
                                    return None, "화자 벡터가 누락되었습니다."
                                if not torch.is_tensor(ref_spk):
                                    ref_spk = torch.tensor(ref_spk)

                                items.append(
                                    VoiceClonePromptItem(
                                        ref_code=ref_code,
                                        ref_spk_embedding=ref_spk,
                                        x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                                        icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                                        ref_text=d.get("ref_text", None),
                                    )
                                )

                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()
                            wavs, sr = tts.generate_voice_clone(
                                text=text.strip(),
                                language=language,
                                voice_clone_prompt=items,
                                **kwargs,
                            )
                            return _wav_to_gradio_audio(wavs[0], sr), "음성 생성이 완료되었습니다! ✨"
                        except Exception as e:
                            return None, f"오류 발생: {type(e).__name__}: {e}"

                    save_btn.click(save_prompt, inputs=[ref_audio_s, ref_text_s, xvec_only_s], outputs=[prompt_file_out, err2])
                    gen_btn2.click(load_prompt_and_gen, inputs=[prompt_file_in, text_in2, lang_in2], outputs=[audio_out2, err2])

        gr.HTML(
            """
            <div class="footer-disclaimer">
              <b>안내사항:</b> 본 데모는 Alibaba Qwen3-TTS 신경망 음성 생성 모델의 연구 및 기능 시연을 위해 제공됩니다.
              생성된 음성은 AI에 의해 실시간 생성되며, 불법적이거나 타인의 권리를 침해하는 용도로의 사용은 엄격히 금지됩니다.
            </div>
            """
        )

    return demo


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.checkpoint and not args.checkpoint_pos:
        parser.print_help()
        return 0

    ckpt = _resolve_checkpoint(args)

    dtype = _dtype_from_str(args.dtype)
    attn_impl = "flash_attention_2" if args.flash_attn else None

    tts = Qwen3TTSModel.from_pretrained(
        ckpt,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    gen_kwargs_default = _collect_gen_kwargs(args)
    demo = build_demo(tts, ckpt, gen_kwargs_default)

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        ssl_verify=True if args.ssl_verify else False,
    )
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile
    if getattr(demo, "theme", None) is not None:
        launch_kwargs["theme"] = demo.theme
    launch_kwargs["css"] = CUSTOM_CSS
    launch_kwargs["head"] = HEAD_HTML
    launch_kwargs["js"] = CUSTOM_JS

    concurrency_limit = 1 if args.device == "mps" or (torch.backends.mps.is_available() and args.device != "cpu") else int(args.concurrency)
    demo.queue(default_concurrency_limit=concurrency_limit).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
