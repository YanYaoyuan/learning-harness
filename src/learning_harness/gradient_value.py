"""Exact LoRA features for the pinned LESS-style optimizer-aware baseline.

No GPU work at import. Features follow the official update expression, while
streaming target means avoids materializing the full candidate gradient store.
"""


def lora_parameters(model):
    pairs = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not pairs or any("lora_" not in name for name, _ in pairs):
        raise ValueError("Only LoRA parameters may require gradients")
    return pairs


def flat_gradient(pairs):
    import torch

    if any(p.grad is None for _, p in pairs):
        raise ValueError("A trainable LoRA gradient is missing")
    vector = torch.cat([p.grad.detach().reshape(-1).float() for _, p in pairs])
    if not torch.isfinite(vector).all():
        raise ValueError("Nonfinite gradient")
    return vector


def optimizer_moments(optimizer, pairs):
    import torch

    for _, p in pairs:
        state = optimizer.state[p]
        if not {"exp_avg", "exp_avg_sq", "step"} <= set(state):
            raise ValueError("Warmup optimizer state is incomplete")
        if state["step"].item() <= 0:
            raise ValueError("Optimizer has not taken an update")
    first, second = [
        torch.cat(
            [optimizer.state[p][key].detach().reshape(-1).float() for _, p in pairs]
        )
        for key in ("exp_avg", "exp_avg_sq")
    ]
    return first, second


def adam_feature(gradient, first, second, beta1=0.9, beta2=0.999, epsilon=1e-8):
    import torch

    if (
        gradient.ndim != 1
        or gradient.shape != first.shape
        or first.shape != second.shape
        or not 0 <= beta1 < 1
        or not 0 <= beta2 < 1
        or epsilon <= 0
        or not all(torch.isfinite(v).all() for v in (gradient, first, second))
        or (second < 0).any()
    ):
        raise ValueError("Invalid Adam feature inputs")
    # This is the official LESS expression, including epsilon inside the root.
    # It is a feature with a frozen state, not an optimizer.step() or a claim
    # to reproduce PyTorch's bias-corrected AdamW parameter update exactly.
    return (beta1 * first + (1 - beta1) * gradient) / (
        beta2 * second + (1 - beta2) * gradient.square() + epsilon
    ).sqrt()


def unit(vector):
    import torch

    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Zero or nonfinite feature norm")
    return vector / norm


def streaming_target_means(gradients):
    """Mean of individually normalized support gradients within each task family."""
    sums, counts = {}, {}
    for group, gradient in gradients:
        value = unit(gradient)
        if group not in sums:
            sums[group] = value.clone()
            counts[group] = 1
        else:
            sums[group].add_(value)
            counts[group] += 1
    if not sums:
        raise ValueError("Empty target support")
    return {k: value / counts[k] for k, value in sums.items()}, counts


def combine_scores(checkpoints, weights):
    import numpy as np

    values, weights = (
        np.asarray(checkpoints, dtype=float),
        np.asarray(weights, dtype=float),
    )
    if (
        values.ndim != 3
        or weights.shape != (values.shape[0],)
        or values.shape[2] != 2
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError(
            "Finite checkpoint/candidate/two-family scores and positive weights required"
        )
    weighted = (values * (weights / weights.sum())[:, None, None]).sum(axis=0)
    # Match the official mean-over-support / max-over-task aggregation order:
    # checkpoints must be combined BEFORE the maximum over target families.
    return weighted.max(axis=1).tolist()
