"""Open-loop dream rollouts with per-step hallucination signals and ground truth.

For each window of real frames + actions:
  * the first `ctx` real frames are encoded and given as context,
  * the world model then dreams `horizon` frames open-loop, conditioned on the
    REAL actions, sliding a context window of at most `ctx_window` latents
    (the model was trained with T = 24),
  * at every dreamed step we log the paper's three label-free signals and
    the true error against the real frame.

Per-step outputs (arrays of shape (n_windows, horizon)):
  u_f        flow instability of the dream sample (seed 0)
  u_s        inter-seed variance across n_seeds denoising samples
  u_s        variance across seeds 0 and 1 (paper: n_samples_unc = 2)
  u_s4       variance across all n_seeds
  u_r        tokenizer round-trip residual of the mean of seeds 0 and 1 (paper: URNormScorer)
  u_r_s0     the same for the seed-0 prediction that the dream follows
  motion     RMS change from the previous latent to the mean of seeds 0 and 1 (paper)
  motion_real  RMS change between consecutive REAL latents (for the paper's dataset-average m)
  lat_err    RMS(z_dream - z_real)
  mse, mse_floor   pixel MSE of the dream frame / of repeating the last
                   context frame, both against the real frame
  dpsnr_raw  10*log10(mse_floor / mse), exactly as the paper (inf when the scene is static)
  dpsnr      10*log10((mse_floor + c) / (mse + c)), c = MSE_NOISE_FLOOR; <= 0 means the dream is no better
             than a frozen frame (the paper's scene-divergence criterion,
             here per step instead of per sequence)

The dream itself follows seed 0. The other seeds are only used for u_s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from .sampler import sample_next_latent


@dataclass
class WM:
    encoder: object
    decoder: object
    dyn: object
    k_max: int
    n_spatial: int
    d_spatial: int
    d_bottleneck: int
    patch: int
    packing_factor: int
    sched: Dict
    lang_dim: int
    device: torch.device
    H: int = 224
    W: int = 224
    C: int = 3


def load_wm(tok_ckpt: str, dyn_ckpt: str, *, device, eval_d: float = 0.25, packing_factor: int = 2) -> WM:
    # eval_d = 0.25 (K = 4 denoising steps) as in the paper's evaluation code (collect_data.py,
    # train_dynamics.py); the live demo uses 0.125.
    # Upstream loaders; requires mmbench2/src on sys.path.
    from interactive import load_tokenizer_from_ckpt, load_dynamics_from_ckpt, make_tau_schedule

    tok, info = load_tokenizer_from_ckpt(tok_ckpt, device)
    dyn, _, _, dinfo = load_dynamics_from_ckpt(
        dyn_ckpt, device=device, d_bottleneck=int(info["d_bottleneck"]),
        n_latents=int(info["n_latents"]), packing_factor=packing_factor,
    )
    sched = make_tau_schedule(k_max=int(dinfo["k_max"]), schedule="shortcut", d=eval_d)
    return WM(
        encoder=tok.encoder, decoder=tok.decoder, dyn=dyn, k_max=int(dinfo["k_max"]),
        n_spatial=int(dinfo["n_spatial"]), d_spatial=int(dinfo["d_spatial"]),
        d_bottleneck=int(info["d_bottleneck"]), patch=int(info["patch"]),
        packing_factor=packing_factor, sched=sched, lang_dim=int(dinfo["lang_dim"]),
        device=device, H=int(info["H"]), W=int(info["W"]), C=int(info["C"]),
    )


@torch.inference_mode()
def encode(wm: WM, frames01: torch.Tensor) -> torch.Tensor:
    """frames01 (B, T, C, H, W) in [0,1] -> packed latents (B, T, Sz, Dz) float32."""
    from model import temporal_patchify
    from interactive import pack_bottleneck_to_spatial

    patches = temporal_patchify(frames01, wm.patch)
    with torch.autocast(device_type=wm.device.type, enabled=wm.device.type == "cuda", dtype=torch.bfloat16):
        z, _ = wm.encoder(patches)
    return pack_bottleneck_to_spatial(z, n_spatial=wm.n_spatial, k=wm.packing_factor).float()


@torch.inference_mode()
def decode(wm: WM, z: torch.Tensor) -> torch.Tensor:
    """packed latents (B, Sz, Dz) -> frames (B, C, H, W) in [0,1]."""
    from model import temporal_unpatchify
    from interactive import unpack_spatial_to_bottleneck

    zb = unpack_spatial_to_bottleneck(z.unsqueeze(1), k=wm.packing_factor, d_bottleneck=wm.d_bottleneck)
    with torch.autocast(device_type=wm.device.type, enabled=wm.device.type == "cuda", dtype=torch.bfloat16):
        patches = wm.decoder(zb)
    return temporal_unpatchify(patches.float(), wm.H, wm.W, wm.C, wm.patch)[:, 0].clamp(0, 1)


MSE_NOISE_FLOOR = 1e-4   # pixel MSE on [0, 1] images, i.e. 40 dB PSNR: invisible differences.
# Without it, a scene that stays perfectly still makes mse_floor = 0 and dPSNR = -inf,
# so even a perfect dream of a still scene would count as broken.


def dpsnr_from_mse(mse, mse_floor, c: float = MSE_NOISE_FLOOR):
    """Works on numpy arrays or torch tensors."""
    import numpy as _np
    if isinstance(mse, torch.Tensor):
        return 10 * torch.log10((mse_floor + c) / (mse + c))
    return 10 * _np.log10((_np.asarray(mse_floor) + c) / (_np.asarray(mse) + c))


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).flatten(1).mean(1).sqrt()


@torch.inference_mode()
def dream_batch(
    wm: WM,
    frames_u8: torch.Tensor,        # (B, L, 3, H, W) uint8, L = ctx + horizon
    actions: torch.Tensor,          # (B, L, 16), backward convention
    act_mask: torch.Tensor,         # (B, 16)
    lang_emb: Optional[torch.Tensor],  # (B, lang_dim) or None
    *,
    ctx: int = 8,
    ctx_window: int = 24,
    n_seeds: int = 4,
    tau_ctx: float = 0.1,
    seed: int = 0,
    keep_frames: int = 0,           # keep dreamed frames for the first `keep_frames` windows
    progress=None,                  # optional callable(step_done, total_steps)
) -> Dict[str, np.ndarray]:
    dev = wm.device
    B, L = frames_u8.shape[:2]
    Hz = L - ctx
    N = n_seeds
    gen = torch.Generator(device=dev).manual_seed(seed)

    real = frames_u8.to(dev).float() / 255.0                       # (B, L, C, H, W)
    z_real = encode(wm, real)                                      # (B, L, Sz, Dz)
    acts = actions.to(dev).float()
    mask16 = act_mask.to(dev).float()
    lang = None if lang_emb is None else lang_emb.to(dev).float()
    last_ctx = real[:, ctx - 1]                                    # floor baseline frame

    hist = [z_real[:, k] for k in range(ctx)]                      # dream history, list of (B, Sz, Dz)
    keys = ["u_f", "u_s", "u_s4", "u_r", "u_r_s0", "motion", "motion_real", "lat_err", "mse", "mse_floor",
            "dpsnr", "dpsnr_raw", "u_f_all_seeds"]
    assert n_seeds >= 2, "the paper's u_s and u_r need at least 2 seeds"
    out = {k: np.zeros((B, Hz), dtype=np.float32) for k in keys}
    keep = int(keep_frames)
    dream_frames = [] if keep > 0 else None

    for h in range(Hz):
        g = ctx + h                                                # index of the frame being dreamed
        s = max(0, g - ctx_window)
        past = torch.stack(hist[s:g], dim=1)                       # (B, t, Sz, Dz)
        t = past.shape[1]
        a = acts[:, s:g + 1].clone()                               # (B, t+1, 16)
        m = mask16[:, None, :].expand(B, t + 1, 16).clone()
        a[:, 0] = 0.0                                              # training convention:
        m[:, 0] = 0.0                                              # first context frame has no action

        # replicate each rollout N times (seed-major inside each rollout)
        rep = lambda x: x.repeat_interleave(N, dim=0)
        z_next, u_f = sample_next_latent(
            wm.dyn, past_packed=rep(past), k_max=wm.k_max, sched=wm.sched,
            actions=rep(a), act_mask=rep(m), lang_emb=None if lang is None else rep(lang),
            tau_ctx=tau_ctx, generator=gen,
        )
        z_next = z_next.float().view(B, N, wm.n_spatial, wm.d_spatial)
        u_f = u_f.view(B, N)
        z_dream = z_next[:, 0]                                     # the dream follows seed 0

        z_mean = z_next[:, :2].mean(dim=1)                         # paper: u_r on the mean of 2 seeds
        f_mean = decode(wm, z_mean)
        z_rt_mean = encode(wm, f_mean[:, None])[:, 0]
        f_dream = decode(wm, z_dream)                              # (B, C, H, W)
        z_rt = encode(wm, f_dream[:, None])[:, 0]                  # round trip of the dream itself
        gt = real[:, g]

        out["u_f"][:, h] = u_f[:, 0].cpu().numpy()
        out["u_f_all_seeds"][:, h] = u_f.mean(1).cpu().numpy()
        out["u_s"][:, h] = z_next[:, :2].var(dim=1).flatten(1).mean(1).cpu().numpy()
        out["u_s4"][:, h] = z_next.var(dim=1).flatten(1).mean(1).cpu().numpy()
        out["motion_real"][:, h] = _rms(z_real[:, g] - z_real[:, g - 1]).cpu().numpy()
        out["u_r"][:, h] = _rms(z_mean - z_rt_mean).cpu().numpy()
        out["u_r_s0"][:, h] = _rms(z_dream - z_rt).cpu().numpy()
        out["motion"][:, h] = _rms(z_mean - hist[-1]).cpu().numpy()
        out["lat_err"][:, h] = _rms(z_dream - z_real[:, g]).cpu().numpy()
        mse = (f_dream - gt).pow(2).flatten(1).mean(1)
        mse_floor = (last_ctx - gt).pow(2).flatten(1).mean(1)
        out["mse"][:, h] = mse.cpu().numpy()
        out["mse_floor"][:, h] = mse_floor.cpu().numpy()
        out["dpsnr"][:, h] = dpsnr_from_mse(mse, mse_floor).cpu().numpy()
        out["dpsnr_raw"][:, h] = (10 * torch.log10(mse_floor / mse.clamp_min(1e-12))).cpu().numpy()

        hist.append(z_dream)
        if progress is not None:
            progress(h + 1, Hz)
        if keep > 0:
            dream_frames.append((f_dream[:keep] * 255).round().to(torch.uint8).cpu())

    if keep > 0:
        out["dream_frames"] = torch.stack(dream_frames, dim=1).numpy()   # (keep, Hz, C, H, W)
    # latent of the last real context frame: the dream's start state (for distance-to-data in Phase 3)
    out["z_start"] = z_real[:, ctx - 1].to(torch.float16).cpu().numpy()
    return out
