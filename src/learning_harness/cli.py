from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lh")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data-root", default="data/alfworld")
    prepare.add_argument("--out", default="data/manifests/alfworld")
    prepare.add_argument("--per-type", type=int, default=10)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--data-root", default="data/alfworld")
    smoke.add_argument("--manifest", default="data/manifests/alfworld/pilot.jsonl")
    teacher = commands.add_parser("build-teacher")
    teacher.add_argument("--data-root", default="data/alfworld")
    teacher.add_argument("--manifest", required=True)
    teacher.add_argument("--config", default="configs/teacher.example.json")
    teacher.add_argument("--out", required=True)
    teacher.add_argument("--ledger", default="runs/teacher-budget.json")
    run = commands.add_parser("rollout")
    run.add_argument("--config", default="configs/pilot.json")
    run.add_argument("--data-root", default="data/alfworld")
    run.add_argument("--out", required=True)
    run.add_argument("--evaluation", action="store_true")
    run.add_argument("--limit", type=int)
    summary = commands.add_parser("summarize")
    summary.add_argument("run_dir")
    export = commands.add_parser("export-sft")
    export.add_argument("run_dir")
    export.add_argument("--arm", required=True)
    export.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        packages = {}
        for name in ("alfworld", "textworld", "numpy", "torch", "transformers", "peft"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        result = dict(python=sys.version.split()[0], packages=packages,
                      visible_gpu_devices=[str(p) for p in Path("/dev").glob("nvidia[0-9]*")],
                      student_endpoint_configured=bool(os.environ.get("LH_STUDENT_BASE_URL")),
                      teacher_endpoint_configured=bool(os.environ.get("LH_TEACHER_BASE_URL")),
                      teacher_key_configured=bool(os.environ.get("LH_TEACHER_API_KEY")))
    elif args.command == "prepare":
        from .manifests import prepare
        result = prepare(args.data_root, args.out, args.per_type)
    elif args.command == "smoke":
        from .common import digest, game_path, read_jsonl
        from .environment import AlfWorld
        tasks = read_jsonl(args.manifest)
        by_type = {}
        for task in tasks:
            by_type.setdefault(task["task_type"], task)
        traces = []
        for task in by_type.values():
            observations = []
            for _ in range(2):
                env = AlfWorld(game_path(args.data_root, task), 11, 5)
                try:
                    initial = env.reset()
                    next_observation, _, _ = env.step("look")
                    observations.append((initial, next_observation))
                finally:
                    env.close()
            if observations[0] != observations[1]:
                raise ValueError("Environment reset/replay is not deterministic")
            traces.append(dict(task_id=task["task_id"], reset_sha=digest(observations[0][0]),
                               replay_identical=True, initial_preview=observations[0][0][:180]))
        result = dict(kind="real_ALFWorld_environment_check_not_model_evaluation", traces=traces)
    elif args.command == "build-teacher":
        from .teacher import build_teacher
        result = build_teacher(args.manifest, args.data_root, args.config, args.out, args.ledger)
    elif args.command == "rollout":
        from .runner import rollout
        result = rollout(args.config, args.data_root, args.out, args.evaluation, args.limit)
    elif args.command == "summarize":
        from .analysis import summarize
        result = summarize(args.run_dir)
    else:
        from .analysis import export_sft
        result = export_sft(args.run_dir, args.arm, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

