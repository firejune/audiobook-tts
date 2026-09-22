# coding=utf-8
# Copyright 2026 The Audiobook-TTS Authors.
# SPDX-License-Identifier: Apache-2.0
"""
Heterogeneous Multi-Device Worker Pool and Execution Engines.
Supports simultaneous allocation across multiple discrete GPUs (e.g. CUDA:1, CUDA:0)
and multi-core Host CPUs with zero IPC lock contention.
"""

import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from qwen_tts import Qwen3TTSModel


class StopOnFlagCriteria(StoppingCriteria):
    """Stopping criteria that halts generation when a stop signal is triggered."""
    def __init__(self):
        super().__init__()
        self.stop = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return bool(self.stop)


SPEAKER_ALIASES = {
    "uncle": "uncle_fu",
    "uncle fu": "uncle_fu",
    "uncle_fu": "uncle_fu",
    "anna": "ono_anna",
    "ono anna": "ono_anna",
    "ono_anna": "ono_anna",
}


def normalize_speaker(speaker: str) -> str:
    """Normalize user-friendly speaker names to model config speaker IDs."""
    if not speaker:
        return "vivian"
    s = str(speaker).strip().lower()
    return SPEAKER_ALIASES.get(s, s)


class WorkerEngine:
    """
    Dedicated execution engine managing a single physical compute device (CUDA GPU, MPS, or CPU).
    Guarantees thread-safe inference execution with per-device synchronization.
    """
    def __init__(self, device: str, checkpoint: str, dtype: torch.dtype):
        self.device = device
        self.lock = threading.Lock()
        self.current_stop_crit: Optional[StopOnFlagCriteria] = None
        print(f"[Worker] Initializing engine on device: {device} (dtype: {dtype})...")
        self.model = Qwen3TTSModel.from_pretrained(
            checkpoint,
            device_map=device,
            torch_dtype=dtype,
        )
        print(f"[Worker] Device {device} ready for generation.")

    def infer(self, text: str, language: str, speaker: str, instruct: Optional[str] = None) -> Tuple[List[np.ndarray], int]:
        """Perform neural speech generation for a single sentence unit."""
        stop_crit = StopOnFlagCriteria()
        self.current_stop_crit = stop_crit
        kwargs: Dict[str, Any] = {
            "stopping_criteria": StoppingCriteriaList([stop_crit])
        }
        resolved_speaker = normalize_speaker(speaker)
        with self.lock:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=resolved_speaker,
                instruct=(instruct or "").strip() or None,
                non_streaming_mode=False,
                **kwargs,
            )
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            elif torch.backends.mps.is_available():
                torch.mps.synchronize()
            self.current_stop_crit = None
            return wavs, sr

    def trigger_stop(self):
        """Immediately abort active generation on this worker."""
        if self.current_stop_crit is not None:
            self.current_stop_crit.stop = True


class WorkerPool:
    """
    Heterogeneous hardware scheduler that pools multiple disparate compute units
    (discrete GPUs, integrated GPUs, host CPU threads) to saturate generation throughput.
    """
    def __init__(self, devices: List[str], checkpoint: str, dtype: torch.dtype):
        self.devices = devices
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.workers: List[WorkerEngine] = [
            WorkerEngine(dev, checkpoint, dtype) for dev in devices
        ]
        self.idle_queue: asyncio.Queue[WorkerEngine] = asyncio.Queue()
        for w in self.workers:
            self.idle_queue.put_nowait(w)

    def trigger_stop_all(self):
        """Broadcast abort signal to all worker engines in the pool."""
        for w in self.workers:
            w.trigger_stop()

    def get_info(self) -> Dict[str, Any]:
        """Return aggregate pool status, device names, and supported voice speakers."""
        if not self.workers:
            return {}
        w0 = self.workers[0]
        model_speakers = w0.model.get_supported_speakers()
        if model_speakers:
            name_map = {
                "vivian": "Vivian",
                "serena": "Serena",
                "sohee": "Sohee",
                "ryan": "Ryan",
                "aiden": "Aiden",
                "ono_anna": "Ono Anna",
                "uncle_fu": "Uncle",
                "eric": "Eric",
                "dylan": "Dylan",
            }
            speakers = [name_map.get(s.lower(), s.title()) for s in model_speakers]
        else:
            speakers = ["Vivian", "Serena", "Sohee", "Ryan", "Aiden", "Ono Anna", "Uncle", "Eric", "Dylan"]
        languages = getattr(w0.model.model, "languages", []) or ["English", "Korean", "Chinese", "Japanese", "Auto"]
        model_type = getattr(w0.model.model, "tts_model_type", "custom_voice")
        device_names = []
        for dev in self.devices:
            if dev.startswith("cuda") and torch.cuda.is_available():
                idx = int(dev.split(":")[1]) if ":" in dev else 0
                device_names.append(f"{dev} ({torch.cuda.get_device_name(idx)})")
            elif dev == "cpu":
                device_names.append("cpu (Host CPU)")
            elif dev == "mps":
                device_names.append("mps (Apple Silicon)")
            else:
                device_names.append(dev)
        return {
            "checkpoint": self.checkpoint,
            "devices": self.devices,
            "device": " + ".join(device_names),
            "model_type": model_type,
            "speakers": speakers,
            "languages": languages,
            "worker_count": len(self.workers),
        }
