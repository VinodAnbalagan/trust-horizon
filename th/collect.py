"""Collect dreams for one task and save them as one .npz file.

Used by phase1.py. One call = one task = one file, so a long run can stop at
any point and resume by skipping tasks whose file already exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def _thumbs(frames_u8: np.ndarray, size: int = 64) -> np.ndarray:
    """(T, 3, H, W) uint8 -> (T, 3, size, size) uint8, area downsampling."""
    x = torch.from_numpy(frames_u8).float()
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="area")
    return x.round().clamp(0, 255).to(torch.uint8).numpy()


def choose_horizon(task: str, tasks_json: str, ctx: int, max_horizon: int, min_horizon: int = 16) -> int:
    with open(tasks_json) as f:
        meta = json.load(f)
    ep = int(meta.get(task, {}).get("max_episode_steps", 10_000))
    return max(min_horizon, min(max_horizon, ep - ctx))


def collect_task(
    wm,
    task: str,
    partition: str,
    out_path: Path,
    *,
    tasks_json: str,
    cache_dir: str,
    ctx: int,
    horizon: int,
    n_windows: int,
    n_seeds: int,
    tau_ctx: float,
    seed: int,
    batch_windows: int = 8,
    n_thumb_windows: int = 1,
    progress=None,
) -> Dict:
    """Dream on `n_windows` windows of `task`, save arrays to out_path. Returns a small meta dict."""
    from th.data import fetch_task_files, load_windows, task_lang_emb
    from th.rollout import dream_batch

    t0 = time.time()
    paths = fetch_task_files(task, partition, cache_dir=cache_dir)
    root = os.path.dirname(paths[0])

    # Longest horizon <= requested that the task's episode length allows, then shrink
    # further only if the recorded episodes are shorter (early termination).
    h = choose_horizon(task, tasks_json, ctx, horizon)
    wins = load_windows(task, partition, root, length=ctx + h, n_windows=n_windows, tasks_json=tasks_json, seed=seed)
    while not wins and h > 16:
        h = max(16, h // 2)
        wins = load_windows(task, partition, root, length=ctx + h, n_windows=n_windows, tasks_json=tasks_json, seed=seed)
    t_data = time.time() - t0
    if not wins:
        return {"task": task, "skipped": f"no episode with >= {ctx + 16} frames"}

    le = task_lang_emb(task, tasks_json, wm.lang_dim)
    chunks = []
    t1 = time.time()
    for i in range(0, len(wins), batch_windows):
        part = wins[i:i + batch_windows]
        frames = torch.stack([w.frames for w in part])
        acts = torch.stack([w.actions for w in part])
        mask = torch.stack([w.act_mask for w in part])
        lang = None if le is None else le.unsqueeze(0).expand(len(part), -1)
        keep = max(0, n_thumb_windows - i)
        out = dream_batch(wm, frames, acts, mask, lang, ctx=ctx, n_seeds=n_seeds, tau_ctx=tau_ctx,
                          seed=seed + i, keep_frames=min(keep, len(part)), progress=progress)
        if "dream_frames" in out:
            d = out.pop("dream_frames")
            out["thumb_dream"] = np.stack([_thumbs(x) for x in d])
            out["thumb_real"] = np.stack([_thumbs(f[ctx:].numpy()) for f in frames[: d.shape[0]]])
        chunks.append(out)
    t_dream = time.time() - t1

    arrays = {k: np.concatenate([c[k] for c in chunks if k in c]) for k in chunks[0]}
    arrays["start"] = np.array([w.start for w in wins])
    arrays["episode"] = np.array([w.episode for w in wins])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")   # write then rename: no half files on crash
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)

    return {
        "task": task, "partition": partition, "n_windows": len(wins), "horizon": h, "ctx": ctx,
        "n_seeds": n_seeds, "tau_ctx": tau_ctx, "seed": seed,
        "secs_data": round(t_data, 1), "secs_dream": round(t_dream, 1),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
