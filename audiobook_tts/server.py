# coding=utf-8
# Copyright 2026 The Audiobook-TTS Authors.
# SPDX-License-Identifier: Apache-2.0
"""
FastAPI streaming server and CLI entrypoint for Audiobook-TTS Studio.
Provides real-time Server-Sent Events (SSE) streaming and mounts the Web Studio UI.
"""

import argparse
import asyncio
import gc
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .pool import WorkerPool
from .text import split_into_sentences, wav_to_base64_data_url

# Root paths
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
WEB_DIR = REPO_ROOT / "web"
SETTINGS_FILE = REPO_ROOT / "last_settings.json"

# Concurrency & Stop Flags
_STOP_REQUESTED = False
_CURRENT_GEN_ID = 0
global_pool: Optional[WorkerPool] = None


def get_worker_pool() -> WorkerPool:
    global global_pool
    if global_pool is None:
        raise RuntimeError("Worker pool is not initialized.")
    return global_pool


def _load_settings() -> Dict[str, Any]:
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load settings: {e}")
    return {}


def _save_settings(data: Dict[str, Any]):
    try:
        cur = _load_settings()
        cur.update(data)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to save settings: {e}")


app = FastAPI(
    title="Audiobook-TTS Studio",
    description="Heterogeneous Multi-Device Streaming Audiobook Generator",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    text: str
    language: str = "English"
    speaker: str = "Vivian"
    instruct: Optional[str] = None
    stream: bool = True


@app.get("/")
async def index():
    html_path = WEB_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Web Studio index.html not found.")
    return FileResponse(html_path, media_type="text/html")


@app.get("/api/info")
async def api_info():
    pool = get_worker_pool()
    return pool.get_info()


@app.get("/api/settings")
async def api_get_settings():
    return _load_settings()


@app.post("/api/settings")
async def api_save_settings(req: Request):
    data = await req.json()
    _save_settings(data)
    return {"status": "ok"}


@app.post("/api/generate/stop")
async def stop_generation():
    global _STOP_REQUESTED, _CURRENT_GEN_ID
    _STOP_REQUESTED = True
    _CURRENT_GEN_ID += 1
    pool = get_worker_pool()
    pool.trigger_stop_all()
    print("[Server] Abort signal dispatched across all active workers.")
    return {"status": "ok"}


@app.post("/api/generate/stream")
async def generate_stream(req: GenerateRequest):
    global _STOP_REQUESTED, _CURRENT_GEN_ID
    _CURRENT_GEN_ID += 1
    my_gen_id = _CURRENT_GEN_ID
    _STOP_REQUESTED = False

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

    pool = get_worker_pool()
    sentences = split_into_sentences(text) if req.stream else [text]
    total_sents = len(sentences)

    async def event_generator():
        # 1. Initial event announcing full chapter text structure
        start_payload = json.dumps({
            "type": "start",
            "total": total_sents,
            "sentences": sentences,
            "workers": len(pool.workers),
        }, ensure_ascii=False)
        yield f"data: {start_payload}\n\n"

        work_queue: asyncio.Queue = asyncio.Queue()
        for i, s in enumerate(sentences):
            work_queue.put_nowait((i, s))

        out_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        async def worker_runner(worker):
            try:
                while True:
                    if _STOP_REQUESTED or _CURRENT_GEN_ID != my_gen_id:
                        break
                    try:
                        idx, sent = work_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    await out_queue.put({
                        "type": "chunk_start",
                        "index": idx,
                        "text": sent,
                        "device": worker.device,
                    })

                    def _do_infer():
                        return worker.infer(
                            text=sent,
                            language=req.language,
                            speaker=req.speaker,
                            instruct=req.instruct,
                        )

                    wavs, sent_sr = await loop.run_in_executor(None, _do_infer)

                    if _STOP_REQUESTED or _CURRENT_GEN_ID != my_gen_id:
                        work_queue.task_done()
                        break

                    if wavs and len(wavs) > 0:
                        b64_url = wav_to_base64_data_url(wavs[0], sent_sr)
                        await out_queue.put({
                            "type": "chunk",
                            "index": idx,
                            "total": total_sents,
                            "text": sent,
                            "audio_url": b64_url,
                            "device": worker.device,
                        })
                    work_queue.task_done()
            except Exception as e:
                print(f"[Worker Error on {worker.device}] {e}")
            finally:
                pool.idle_queue.put_nowait(worker)

        # Launch concurrent runners for all workers in pool
        worker_tasks = [asyncio.create_task(worker_runner(w)) for w in pool.workers]

        async def supervisor():
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            await out_queue.put(None)

        asyncio.create_task(supervisor())

        completed_chunks = 0
        while True:
            item = await out_queue.get()
            if item is None:
                break
            if _STOP_REQUESTED or _CURRENT_GEN_ID != my_gen_id:
                yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                break

            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            if item.get("type") == "chunk":
                completed_chunks += 1

        if not (_STOP_REQUESTED or _CURRENT_GEN_ID != my_gen_id):
            yield f"data: {json.dumps({'type': 'done', 'total_generated': completed_chunks})}\n\n"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def init_server(checkpoint: str, devices: List[str], dtype_str: str = "bfloat16"):
    global global_pool

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str.lower(), torch.bfloat16)

    print("=" * 60)
    print(f"Initializing Audiobook-TTS Studio Server")
    print(f"Checkpoint : {checkpoint}")
    print(f"Devices    : {', '.join(devices)} ({len(devices)} parallel workers)")
    print(f"Dtype      : {dtype}")
    print("=" * 60)

    pool = WorkerPool(devices, checkpoint, dtype)
    global_pool = pool
    print(f"All {len(devices)} workers active and waiting for requests.\n")


def main():
    parser = argparse.ArgumentParser(description="Audiobook-TTS Streaming Studio Server")
    parser.add_argument("checkpoint_pos", nargs="?", default=None, help="Model checkpoint path or HF repo")
    parser.add_argument("-c", "--checkpoint", default=None, help="Model checkpoint path or HF repo")
    parser.add_argument("--device", default=None, help="Single compute device (e.g. mps, cuda:0, cpu)")
    parser.add_argument("--devices", default=None, help="Comma-separated compute devices for worker pool (e.g. cuda:1,cuda:0,cpu)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--ip", default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    # Determine devices
    if args.devices:
        dev_list = [d.strip() for d in args.devices.split(",") if d.strip()]
    elif args.device:
        dev_list = [args.device.strip()]
    else:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count >= 2:
                dev_list = [f"cuda:{i}" for i in range(count)]
            else:
                dev_list = ["cuda:0"]
        elif torch.backends.mps.is_available():
            dev_list = ["mps"]
        else:
            dev_list = ["cpu"]

    ckpt = args.checkpoint or args.checkpoint_pos or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    init_server(ckpt, devices=dev_list, dtype_str=args.dtype)

    print(f"🚀 Audiobook-TTS Studio running at: http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.ip, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
