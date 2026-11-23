"""Batched next-latent sampler that reports flow instability per sample.

Adapted from `sample_one_timestep_packed` in nicklashansen/mmbench2
(src/interactive.py, MIT license, commit 3dda6ea). The only behavioral change:
the upstream function averages the flow-instability signal (u_f) over the
whole batch, which is fine for one interactive session but mixes windows and
seeds together when we batch. Here u_f is returned per batch element, so one
forward pass can serve many rollouts and many noise seeds at once.

Everything else (shortcut schedule, context corruption tau_ctx, action convention)
follows upstream. Precision is bfloat16, as in training and in the paper's evaluation
code (train_dynamics.sample_one_timestep_packed); the interactive demo we first copied
used float16 autocast, which the first Phase 1 repro run inherited.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch


@torch.inference_mode()
def sample_next_latent(
    dyn,
    *,
    past_packed: torch.Tensor,            # (B, t, n_spatial, d_spatial)
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor],      # (B, t+1, A); actions[:, k] produced frame k
    act_mask: Optional[torch.Tensor],     # (B, t+1, A)
    lang_emb: Optional[torch.Tensor],     # (B, lang_dim)
    tau_ctx: float = 0.1,
    use_amp: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (z_next (B, n_spatial, d_spatial), u_f (B,))."""
    device = past_packed.device
    dtype = past_packed.dtype
    B, t, n_spatial, d_spatial = past_packed.shape

    K = int(sched["K"])
    e = int(sched["e"])
    tau = sched["tau"]
    tau_idx = sched["tau_idx"]
    dt = float(sched["dt"])
    emax = int(round(math.log2(int(k_max))))

    z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype, generator=generator)

    if tau_ctx > 0.0 and t > 0:
        z0_ctx = torch.randn(past_packed.shape, device=device, dtype=dtype, generator=generator)
        past_input = ((1.0 - tau_ctx) * past_packed.float() + tau_ctx * z0_ctx.float()).to(dtype)
        ctx_sig_idx = min(int(round((1.0 - tau_ctx) * k_max)), k_max)
    else:
        past_input = past_packed
        ctx_sig_idx = k_max

    actions_in = None if actions is None else actions[:, : t + 1]
    actmask_in = None if act_mask is None else act_mask[:, : t + 1]

    x1_prev = None
    deltas = []  # each (B,)
    autocast_on = use_amp and device.type == "cuda"

    for i in range(K):
        sig_i = int(tau_idx[i])
        step_idxs = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
        step_idxs[:, -1] = e
        signal_idxs = torch.full((B, t + 1), ctx_sig_idx, device=device, dtype=torch.long)
        signal_idxs[:, -1] = sig_i
        packed_seq = torch.cat([past_input, z], dim=1)
        with torch.autocast(device_type=device.type, enabled=autocast_on, dtype=torch.bfloat16):
            x1_full, _ = dyn(
                actions_in, step_idxs, signal_idxs, packed_seq,
                act_mask=actmask_in, agent_tokens=None, lang_emb=lang_emb,
            )
        x1 = x1_full[:, -1:].float()
        if x1_prev is not None:
            deltas.append((x1 - x1_prev).pow(2).mean(dim=(1, 2, 3)).sqrt())
        x1_prev = x1
        denom = max(1e-4, 1.0 - float(tau[i]))
        z = (z.float() + (x1 - z.float()) / denom * dt).to(dtype)

    if deltas:
        d = torch.stack(deltas, dim=0)               # (K-1, B)
        tail = d[d.shape[0] // 2:] if d.shape[0] > 1 else d
        u_f = tail.mean(dim=0)
    else:
        u_f = torch.zeros(B, device=device)
    return z[:, 0], u_f
