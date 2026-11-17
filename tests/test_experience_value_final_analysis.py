import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "final_analysis", Path(__file__).parents[1] / "scripts/analyze_experience_value_final.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture_models():
    records = []
    for source in module.SOURCES:
        for policy in module.POLICIES:
            for seed in (1, 2):
                for sample in (10, 20):
                    effect = (0.2 if seed == 1 else -0.1) if source == "api" else 0.0
                    records.append(dict(
                        case=dict(name=f"{source}-{policy}-{seed}-{sample}", source=source,
                                  policy=policy, initialization_seed=seed, sampling_seed=sample),
                        mean_success=0.5 + (effect if policy == "gradient_high" else 0),
                    ))
    return records


def test_sampling_replicates_remain_nested_within_initializations():
    result = module.paired_contrasts(fixture_models(), (1, 2), (10, 20))["random"]
    assert result["equal_source_per_initialization"] == pytest.approx({"1": 0.1, "2": -0.05})
    assert result["mean_difference"] == pytest.approx(0.025)
    assert result["positive_initializations"] == 1
    assert result["negative_initializations"] == 1


def test_incomplete_and_duplicated_matrices_are_rejected():
    models = fixture_models()
    with pytest.raises(ValueError, match="complete crossed matrix"):
        module.paired_contrasts(models[:-1], (1, 2), (10, 20))
    with pytest.raises(ValueError, match="Duplicate"):
        module.paired_contrasts(models + models[:1], (1, 2), (10, 20))
