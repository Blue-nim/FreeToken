"""Torch reference sampling ops, used on architectures where the Triton sampling
kernels cannot compile (Pascal sm_61 and below: Triton 3.6.0 emits atomic ordering
suffixes `.acq_rel`/`.relaxed` that ptxas rejects below `.target sm_70`).

These are exact, numerically-equivalent implementations of the Triton module's
public surface (softmax / top-k / top-p / combined + draw) using only torch ops.
They are NOT performance-tuned -- Pascal is a bring-up / compatibility target, and
the dense 0.5B-class models it runs are far from saturating the sampler. Correctness
first; speed later.

Kept drop-in compatible with ``freetoken.kernel.triton.sampling``'s public API so
``engine/sample.py`` can switch on arch with a one-line import change.
"""

from __future__ import annotations

import torch


def _softmax(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    # logits: [B, V], temperatures: [B] or scalar
    if temperatures.dim() == 0 or temperatures.numel() == 1:
        t = temperatures.reshape(-1)
        if t.numel() == 1:
            t = t.expand(logits.shape[0])
        t = t.reshape(-1, 1)
    else:
        t = temperatures.reshape(-1, 1)
    # guard against zero/eps temperatures
    t = t.clamp_min(1e-6)
    z = logits.float() / t
    return torch.softmax(z, dim=-1)


def softmax(logits: torch.Tensor, temperatures: torch.Tensor, enable_pdl: bool = False):
    # enable_pdl is a Triton-only tuning flag; ignored here.
    return _softmax(logits, temperatures)


def _top_k_renorm(probs: torch.Tensor, top_k: torch.Tensor | int) -> torch.Tensor:
    if isinstance(top_k, torch.Tensor):
        top_k = top_k.to(probs.device)
    else:
        top_k = int(top_k)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    if isinstance(top_k, torch.Tensor):
        k = top_k.to(torch.long).clamp(min=1)
        # k-th value per row
        idx = (k - 1).clamp(min=0).unsqueeze(-1)
        thr = torch.gather(sorted_probs, -1, idx)  # [B,1]
    else:
        k = max(1, min(top_k, probs.shape[-1]))
        thr = sorted_probs[:, k - 1:k]
    kept = probs * (probs >= thr)
    s = kept.sum(dim=-1, keepdim=True)
    s = s.clamp_min(1e-30)
    return kept / s


def top_k_renorm_probs(probs: torch.Tensor, top_k):
    return _top_k_renorm(probs, top_k)


def _top_p_renorm(probs: torch.Tensor, top_p: torch.Tensor | float) -> torch.Tensor:
    if isinstance(top_p, torch.Tensor):
        top_p = top_p.to(probs.device)
    else:
        top_p = float(top_p)
    # sort descending
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # threshold: smallest value whose cumulative sum >= top_p
    if isinstance(top_p, torch.Tensor):
        tp = top_p.reshape(-1, 1)
    else:
        tp = float(top_p)
    mask_sorted = cum - sorted_probs < tp  # keep while cumulative-before < top_p
    # threshold = last kept sorted value
    # use the value where mask flips; simpler: thr = sorted_probs where mask, take max per row
    thr = (sorted_probs * mask_sorted.float()).amax(dim=-1, keepdim=True)
    kept = probs * (probs >= thr)
    s = kept.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return kept / s


def top_p_renorm_probs(probs: torch.Tensor, top_p):
    return _top_p_renorm(probs, top_p)


@torch.no_grad()
def _draw(renormed: torch.Tensor, generator=None, offset=None):
    # multinomial over renormed probs (already zeroed outside keep set)
    B, V = renormed.shape
    # normalize defensively (already sums to ~1 within kept set)
    p = renormed / renormed.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    idx = torch.multinomial(p, num_samples=1, generator=generator)
    return idx.reshape(-1)


def sampling_from_probs(probs: torch.Tensor, generator=None):
    return _draw(probs, generator=generator)


def top_k_sampling_from_probs(probs, top_k, indices=None, deterministic=True,
                              generator=None, check_nan=False, seed=None, offset=None,
                              return_valid=False):
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    r = _top_k_renorm(src, top_k)
    out = _draw(r, generator=generator)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_k_top_p_sampling_from_probs(probs, top_k, top_p, indices=None,
                                    filter_apply_order="top_k_first", deterministic=True,
                                    generator=None, check_nan=False, seed=None, offset=None,
                                    return_valid=False):
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    if filter_apply_order == "top_k_first":
        r = _top_k_renorm(src, top_k)
        r = _top_p_renorm(r, top_p)
    else:
        r = _top_p_renorm(src, top_p)
        r = _top_k_renorm(r, top_k)
    out = _draw(r, generator=generator)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


__all__ = [
    "softmax", "top_k_renorm_probs", "top_p_renorm_probs",
    "sampling_from_probs", "top_k_sampling_from_probs",
    "top_p_sampling_from_probs", "top_k_top_p_sampling_from_probs",
]
