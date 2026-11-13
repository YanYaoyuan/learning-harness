import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from learning_harness.gradient_value import (
    adam_feature,
    combine_scores,
    flat_gradient,
    lora_parameters,
    optimizer_moments,
    streaming_target_means,
    unit,
)


def test_feature_matches_official_less_function_and_keeps_optimizer_state():
    tree = ast.parse(
        Path(
            "third_party/LESS-reference/less/data_selection/collect_grad_reps.py"
        ).read_text()
    )
    definition = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "obtain_gradients_with_adam"
    )
    namespace = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[definition], type_ignores=[]),
            "pinned_LESS_reference",
            "exec",
        ),
        namespace,
    )
    model = torch.nn.Linear(2, 1, bias=False).double()

    class LossModel(torch.nn.Module):
        def forward(self, x):
            class Result:
                pass

            value = Result()
            value.loss = model(x).square().sum()
            return value

        def named_parameters(self):
            return model.named_parameters()

    first = torch.tensor([0.12, -0.2], dtype=torch.double)
    second = torch.tensor([0.3, 0.01], dtype=torch.double)
    saved = (first.clone(), second.clone())
    reference = namespace[definition.name](
        LossModel(),
        dict(x=torch.tensor([[1.0, 2.0]], dtype=torch.double)),
        first,
        second,
    )
    expected = adam_feature(model.weight.grad.flatten(), first, second)
    assert torch.equal(reference, expected)
    assert torch.equal(first, saved[0]) and torch.equal(second, saved[1])


def test_streaming_similarity_equals_full_candidate_by_support_matrix():
    candidates = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.0]])
    targets = torch.tensor([[0.0, 1.0, 1.0], [2.0, 1.0, 0.0], [1.0, -2.0, 3.0]])
    groups = ["place", "clean", "place"]
    means, counts = streaming_target_means(zip(groups, targets))
    full = (
        torch.nn.functional.normalize(candidates)
        @ torch.nn.functional.normalize(targets).T
    )
    for key in means:
        actual = torch.nn.functional.normalize(candidates) @ means[key]
        expected = full[:, [i for i, k in enumerate(groups) if k == key]].mean(dim=1)
        assert torch.allclose(actual, expected, atol=1e-7)
    assert counts == {"place": 2, "clean": 1}


def test_checkpoint_weighting_precedes_task_maximum():
    values = [[[1, 0]], [[0, 1]]]
    assert combine_scores(values, [1, 1]) == [0.5]
    with pytest.raises(ValueError):
        combine_scores(values, [1, -1])


def test_gradient_matches_finite_difference_on_toy_loss():
    point = torch.tensor([0.3, -0.7], dtype=torch.double, requires_grad=True)
    loss = ((point @ torch.tensor([2.0, -1.0], dtype=torch.double)) - 0.2).square()
    loss.backward()
    epsilon = 1e-6

    def f(x):
        return float((2 * x[0] - x[1] - 0.2) ** 2)

    numeric = np.array(
        [
            (
                f(point.detach().numpy() + epsilon * np.eye(2)[i])
                - f(point.detach().numpy() - epsilon * np.eye(2)[i])
            )
            / (2 * epsilon)
            for i in range(2)
        ]
    )
    assert np.allclose(point.grad.numpy(), numeric, atol=1e-8)


def test_parameters_and_optimizer_order_are_checked():
    model = torch.nn.Module()
    model.register_parameter("lora_a", torch.nn.Parameter(torch.tensor([1.0, 2.0])))
    pairs = lora_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="incomplete"):
        optimizer_moments(optimizer, pairs)
    model.lora_a.square().sum().backward()
    assert torch.equal(flat_gradient(pairs), torch.tensor([2.0, 4.0]))
    optimizer.step()
    first, second = optimizer_moments(optimizer, pairs)
    assert torch.all(first > 0) and torch.all(second > 0)
    with pytest.raises(ValueError, match="Only LoRA"):
        lora_parameters(torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="Zero"):
        unit(torch.zeros(3))
