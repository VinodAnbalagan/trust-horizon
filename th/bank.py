"""Latent "bank" of training data, and distance from a dream's start state to it.

The tokenizer is causal in time, so a frame's latent depends on the frames before it.
A dream's start state (`z_start`) is the latent of the 8th frame of an 8-frame context,
so bank entries are made the same way: 8-frame clips from training trajectories, keeping
the latent of the last frame.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import torch


@torch.inference_mode()
def encode_clips(wm, task: str, partitions: List[str], *, tasks_json: str, cache_dir: str,
                 n_clips: int, ctx: int = 8, batch: int = 32, seed: int = 0) -> np.ndarray:
    """(n, D) float16 latents of the last frame of `n_clips` training clips, spread over partitions."""
    from th.data import fetch_task_files, load_windows
    from th.rollout import encode

    per = max(1, n_clips // len(partitions))
    zs = []
    for part in partitions:
        try:
            paths = fetch_task_files(task, part, cache_dir=cache_dir)
        except FileNotFoundError:
            continue
        wins = load_windows(task, part, os.path.dirname(paths[0]), length=ctx, n_windows=per,
                            tasks_json=tasks_json, seed=seed)
        for i in range(0, len(wins), batch):
            fr = torch.stack([w.frames for w in wins[i:i + batch]]).to(wm.device).float() / 255.0
            z = encode(wm, fr)[:, -1]                                   # (b, Sz, Dz)
            zs.append(z.flatten(1).to(torch.float16).cpu().numpy())
    if not zs:
        raise FileNotFoundError(f"no training data found for {task} in {partitions}")
    return np.concatenate(zs)


def knn_distance(z: np.ndarray, bank: np.ndarray, k: int = 5, chunk: int = 256) -> np.ndarray:
    """Mean L2 distance from each row of z to its k nearest bank rows."""
    B = torch.from_numpy(bank.astype(np.float32))
    out = []
    for i in range(0, len(z), chunk):
        q = torch.from_numpy(z[i:i + chunk].reshape(len(z[i:i + chunk]), -1).astype(np.float32))
        d = torch.cdist(q, B)
        out.append(d.topk(min(k, B.shape[0]), largest=False).values.mean(1).numpy())
    return np.concatenate(out)


def save_bank(path: Path, task: str, bank: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, bank=bank, task=task)
