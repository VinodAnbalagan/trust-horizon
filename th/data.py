"""Load ground-truth trajectories from the raw MMBench2 files.

Raw layout per task and partition (e.g. `val/`):
  {task}.pt          dict with per-frame tensors: "episode", "action", "reward"
  {task}-{i}.png     horizontal strips of 224x224 RGB frames, in order

Action convention (matches upstream training, see train_dynamics.parse_batch):
the stored action at global index i is the action that PRODUCED frame i.
For a window starting at s, actions[k] = stored[s + k], with actions[0] = 0.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

HF_DATASET = "nicklashansen/mmbench2"


@dataclass
class Window:
    task: str
    partition: str
    episode: int
    start: int                 # global frame index of the first context frame
    frames: torch.Tensor       # (L, 3, 224, 224) uint8
    actions: torch.Tensor      # (L, 16) float32, backward convention, actions[0] = 0
    act_mask: torch.Tensor     # (16,) float32


def fetch_task_files(task: str, partition: str, cache_dir: str) -> List[str]:
    """Download {task}.pt and its png strips. Returns local paths."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    prefix = f"{partition}/{task}"
    names = [
        f for f in api.list_repo_files(HF_DATASET, repo_type="dataset")
        if f == f"{prefix}.pt" or (f.startswith(f"{prefix}-") and f.endswith(".png")
                                   and f[len(prefix) + 1:-4].isdigit())
    ]
    if not names:
        raise FileNotFoundError(f"No files for {prefix} in {HF_DATASET}")
    return [hf_hub_download(HF_DATASET, n, repo_type="dataset", cache_dir=cache_dir) for n in names]


def _load_frames(task: str, root: str) -> torch.Tensor:
    from torchvision.io import read_image

    out, i = [], 0
    while True:
        p = os.path.join(root, f"{task}-{i}.png")
        if not os.path.exists(p):
            break
        img = read_image(p)                         # (3, 224, 224*N)
        C, H, W = img.shape
        n = W // 224
        out.append(img.view(C, H, n, 224).permute(2, 0, 1, 3).contiguous())
        i += 1
    if not out:
        raise FileNotFoundError(f"no png strips for {task} in {root}")
    return torch.cat(out, dim=0)


def _meta_key(task: str, meta: dict) -> str:
    """Unseen tasks borrow metadata from a seen partner, as in upstream interactive.py."""
    from interactive import TEST_TASK_SET
    if task in TEST_TASK_SET and TEST_TASK_SET[task] in meta:
        return TEST_TASK_SET[task]
    return task


def task_act_dim(task: str, tasks_json: str) -> int:
    with open(tasks_json) as f:
        meta = json.load(f)
    # Upstream prefers the live env's action space for unseen tasks; offline we use
    # the task's own entry when present, else the borrowed partner's.
    if task in meta:
        return int(meta[task]["action_dim"])
    key = _meta_key(task, meta)
    return int(meta[key]["action_dim"]) if key in meta else 16


def task_lang_emb(task: str, tasks_json: str, lang_dim: int) -> Optional[torch.Tensor]:
    """(lang_dim,) language embedding, or None if the model is unconditioned."""
    if lang_dim <= 0:
        return None
    with open(tasks_json) as f:
        meta = json.load(f)
    key = _meta_key(task, meta)
    te = meta.get(key, {}).get("text_embedding")
    if te is None or len(te) != lang_dim:
        return None
    return torch.tensor(te, dtype=torch.float32)


def load_windows(
    task: str,
    partition: str,
    root: str,
    *,
    length: int,
    n_windows: int,
    tasks_json: str,
    seed: int = 0,
    act_dim: Optional[int] = None,
) -> List[Window]:
    """Sample up to n_windows contiguous windows that stay inside one episode."""
    td = torch.load(os.path.join(root, f"{task}.pt"), map_location="cpu", weights_only=False)
    ep = td["episode"].to(torch.int64).flatten()
    act = td["action"].float()
    if act.ndim == 1:
        act = act.unsqueeze(-1)
    frames = _load_frames(task, root)
    if frames.shape[0] != ep.shape[0]:
        raise RuntimeError(f"{task}: {frames.shape[0]} frames vs {ep.shape[0]} episode labels")

    if act_dim is None:
        act_dim = task_act_dim(task, tasks_json)
    act_dim = max(0, min(16, act_dim))
    mask = torch.zeros(16)
    mask[:act_dim] = 1.0

    # valid starts: window [s, s+length) lies in a single episode
    starts = []
    for e in torch.unique(ep).tolist():
        idx = torch.nonzero(ep == e).flatten()
        lo, hi = int(idx.min()), int(idx.max()) + 1
        starts += [(e, s) for s in range(lo, hi - length + 1)]
    if not starts:
        return []
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(starts), size=min(n_windows, len(starts)), replace=False)

    wins = []
    for j in sorted(pick.tolist()):
        e, s = starts[j]
        a = torch.zeros(length, 16)
        a[:, :act.shape[1]] = torch.nan_to_num(act[s:s + length, :16], nan=0.0)
        a[0] = 0.0
        a = a.clamp(-1, 1) * mask
        wins.append(Window(task, partition, e, s, frames[s:s + length].clone(), a, mask.clone()))
    return wins
