"""Extract numeric training curves from the immutable trainer's fixed log cadence."""

import ast
import math
import re


def training_metrics(text):
    rows = []
    for match in re.finditer(r"\{'loss':[^{}]*\}", text):
        try:
            value = ast.literal_eval(match.group())
        except (ValueError, SyntaxError) as error:
            raise ValueError("Unrecognized training loss record") from error
        fields = {k: value[k] for k in ("loss", "grad_norm", "learning_rate", "epoch")}
        if any(
            type(v) not in (int, float) or not math.isfinite(v) for v in fields.values()
        ):
            raise ValueError("Nonfinite or nonnumeric training telemetry")
        rows.append(dict(step=(len(rows) + 1) * 10, **fields))
    if len(rows) > 19:
        raise ValueError(
            "Unexpected extra/resumed loss records for the frozen 192-step trainer"
        )
    return rows
