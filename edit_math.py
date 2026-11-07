"""Small, deterministic weight-editing primitives (no autograd)."""
from __future__ import annotations
import torch


def contrastive_direction(trigger: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """Difference of mean hidden vectors, normalized to L2=1.

    Inputs have shape [num_examples, hidden_size].
    """
    if trigger.ndim != 2 or control.ndim != 2 or trigger.shape[1] != control.shape[1]:
        raise ValueError("Expected two [N, hidden_size] matrices with equal hidden size")
    if not len(trigger) or not len(control):
        raise ValueError("Calibration sets cannot be empty")
    d = trigger.float().mean(0) - control.float().mean(0)
    norm = torch.linalg.vector_norm(d)
    if norm.item() < 1e-8:
        raise ValueError("Trigger and control means are indistinguishable")
    return d / norm


def edited_row(original: torch.Tensor, direction: torch.Tensor, scale: float) -> torch.Tensor:
    if original.ndim != 1 or direction.shape != original.shape:
        raise ValueError("Output row and edit direction must be 1D and equal length")
    return original + float(scale) * direction.to(original)


def rate(hits: list[bool]) -> float:
    return sum(hits) / len(hits) if hits else float("nan")


def score(hit_trigger: float, hit_control: float, kl_control: float) -> float:
    """Validation objective, larger is better. Only uses validation inputs."""
    return hit_trigger - 1.5 * hit_control - 0.05 * kl_control
