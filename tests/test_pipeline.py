"""CPU test of the rollout pipeline with tiny random models and synthetic data.

Needs the upstream code in third_party/mmbench2 (scripts/setup_runpod.sh puts it there):
    python tests/test_pipeline.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UP = Path(os.environ.get("MMBENCH2", ROOT / "third_party" / "mmbench2")).resolve()
sys.path[:0] = [str(UP / "src"), str(ROOT)]
TASKS_JSON = str(UP / "tasks.json")

from model import Decoder, Dynamics, Encoder, Tokenizer  # noqa: E402
from th.data import load_windows, task_act_dim, task_lang_emb  # noqa: E402
from th.rollout import dream_batch, load_wm  # noqa: E402


def tiny_checkpoints(d):
    n_patches, dp = (224 // 16) ** 2, 16 * 16 * 3
    enc = Encoder(patch_dim=dp, d_model=32, n_latents=16, n_patches=n_patches, n_heads=2, depth=1,
                  d_bottleneck=32, dropout=0.0, mlp_ratio=2.0, time_every=1, mae_p_min=0.0, mae_p_max=0.0)
    dec = Decoder(d_bottleneck=32, d_model=32, n_heads=2, depth=1, n_latents=16, n_patches=n_patches,
                  d_patch=dp, dropout=0.0, mlp_ratio=2.0, time_every=1)
    targs = dict(H=224, W=224, C=3, patch=16, d_model=32, n_heads=2, depth=1, n_latents=16,
                 d_bottleneck=32, mlp_ratio=2.0)
    torch.save({"args": targs, "model": Tokenizer(enc, dec).state_dict()}, f"{d}/tok.pt")
    dyn = Dynamics(d_model=32, d_bottleneck=32, d_spatial=64, n_spatial=8, n_register=2, n_agent=0,
                   n_heads=2, depth=4, k_max=8, dropout=0.0, mlp_ratio=2.0, time_every=4, lang_dim=512)
    with torch.no_grad():  # move off the zero init so the seeds disagree
        for p in dyn.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    dargs = dict(d_model_dyn=32, n_heads=2, dyn_depth=4, k_max=8, n_register=2, n_agent=0,
                 lang_dim=512, mlp_ratio=2.0, time_every=4)
    torch.save({"args": dargs, "model": dyn.state_dict()}, f"{d}/dyn.pt")


def synthetic_task(d, task="walker-walk"):
    from torchvision.io import write_png
    n = 130
    ep = torch.tensor([0] * 60 + [1] * 70)
    torch.save({"episode": ep, "action": torch.rand(n, 6), "reward": torch.rand(n)}, f"{d}/{task}.pt")
    fr = torch.arange(n).view(n, 1, 1, 1).expand(n, 3, 224, 224).to(torch.uint8)  # pixel value = frame index
    for i, (a, b) in enumerate([(0, 50), (50, 130)]):
        write_png(fr[a:b].permute(1, 2, 0, 3).reshape(3, 224, (b - a) * 224), f"{d}/{task}-{i}.png")


def main():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        tiny_checkpoints(d)
        synthetic_task(d)
        wins = load_windows("walker-walk", "val", d, length=40, n_windows=5, tasks_json=TASKS_JSON)
        assert len(wins) == 5
        for w in wins:
            assert int(w.frames[0, 0, 0, 0]) == w.start, "frame alignment"
            assert w.frames.shape[0] == 40
            ep_end = 60 if w.episode == 0 else 130
            assert w.start + 40 <= ep_end, "window crosses an episode boundary"
            assert float(w.actions[0].abs().sum()) == 0.0, "first action must be zero"
            assert int(w.act_mask.sum()) == task_act_dim("walker-walk", TASKS_JSON)

        wm = load_wm(f"{d}/tok.pt", f"{d}/dyn.pt", device=torch.device("cpu"))
        frames = torch.stack([w.frames for w in wins[:2]])
        acts = torch.stack([w.actions for w in wins[:2]])
        mask = torch.stack([w.act_mask for w in wins[:2]])
        lang = task_lang_emb("walker-walk", TASKS_JSON, 512).expand(2, -1)
        a = dream_batch(wm, frames, acts, mask, lang, ctx=8, n_seeds=3, seed=1)
        b = dream_batch(wm, frames, acts, mask, lang, ctx=8, n_seeds=3, seed=1)
        z0 = a.pop("z_start"); b.pop("z_start")
        assert z0.shape[0] == 2 and z0.dtype == np.float16
        for k, v in a.items():
            assert v.shape == (2, 32) and np.isfinite(v).all(), k
            assert np.allclose(v, b[k]), f"{k} not deterministic"
        assert a["u_s"].mean() > 0 and a["u_f"].mean() > 0
        assert not np.allclose(a["u_f"][0], a["u_f"][1]), "u_f should be per window"
    print("all checks passed")


if __name__ == "__main__":
    main()
