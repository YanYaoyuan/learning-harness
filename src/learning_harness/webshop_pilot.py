"""Frozen train-only calibration plan and aggregate-only result recovery."""

import json
import os
from pathlib import Path

from .webshop_factory import stream_sha
from .webshop_requests import summarize_requests
from .webshop_serving import IMPLEMENTATION

PARENT = Path("runs/variance-diagnostic-v1/stage4")
PARENT_SHA = "76f7e74a6c3b9b4da51b8f543f983583c89d954e1abceac2a99ae35dfb48c345"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
IDENTITY_SHA = "ce3818cf416e775fb031cd99a066e2eff017e16333b30d3125c6d2ef38891233"
PROTOCOL = Path("docs/WEBSHOP_DEVELOPMENT_PILOT_PROTOCOL_2026-11-13.zh-CN.md")
REQUIRED_INPUTS = {
    "scripts/run_webshop_pilot.py",
    "scripts/webshop_pilot_worker.py",
    "scripts/serve_webshop_base.py",
    "src/learning_harness/webshop_pilot.py",
    "src/learning_harness/webshop_serving.py",
    "src/learning_harness/webshop_agent.py",
    "src/learning_harness/webshop_requests.py",
    "src/learning_harness/webshop_factory.py",
    "src/learning_harness/webshop_public.py",
    "src/learning_harness/api.py",
    "src/learning_harness/bounded_queue.py",
    "src/learning_harness/common.py",
    "configs/webshop-development-pilot-v1.json",
    str(PROTOCOL),
    "data/models/Qwen3-4B-Instruct-2507/download-manifest.json",
}


def read(path):
    return json.loads(Path(path).read_text())


def validate_config(config):
    exact = dict(
        identity_root="output/webshop-preparation/20261113/goal-identities",
        identity_freeze_sha256=IDENTITY_SHA,
        split="train",
        global_goal_indices=list(range(1500, 1532)),
        prompt_source="third_party/ReAct-WebShop/prompts.json",
        prompt_modes=["prompt1", "prompt1_actonly"],
        student_model="Qwen/Qwen3-4B-Instruct-2507",
        student_revision=REVISION,
        student_adapters=None,
        model_temperature=0,
        rollout_seed=11,
        max_executed_decisions=14,
        max_new_tokens=100,
        stop=["\n"],
        max_context_characters=6400,
        expected_episodes=64,
        teacher_calls=0,
        official_test_evaluations=0,
        eval_episodes=0,
        proposed_gpu_cap_hours=1.0,
        requires_parent_stage4_completed=True,
        requires_dedicated_base_model_server=True,
        existing_model_servers_may_be_queried=False,
        calibration_only=True,
        methods_or_test_protocol_frozen=False,
    )
    if any(config.get(k) != value for k, value in exact.items()):
        raise ValueError("Calibration scope or budget differs from the declared draft")
    # bool/int equality must not admit a semantically changed protocol.
    if any(type(config.get(k)) is not type(v) for k, v in exact.items()):
        raise ValueError("Calibration field types differ")
    return config


def episode_plan(config):
    validate_config(config)
    cases = []
    for index, goal in enumerate(config["global_goal_indices"]):
        modes = config["prompt_modes"]
        # Alternate which prompt goes first, predeclared independently of results.
        for mode in modes if index % 2 == 0 else modes[::-1]:
            cases.append(
                dict(name=f"train-{goal}-{mode}", global_goal_index=goal, mode=mode)
            )
    return cases


def validate_frozen(root):
    root = Path(root)
    frozen = read(root / "freeze.json")
    validate_config(frozen["config"])
    if (
        frozen["cases"] != episode_plan(frozen["config"])
        or frozen["parent_freeze_sha256"] != PARENT_SHA
        or frozen["gpu"] not in (0, 1)
        or type(frozen["gpu"]) is not int
        or frozen["port"] != 19780 + frozen["gpu"]
        or frozen["cap_gpu_seconds"] != 3600
        or frozen["server_implementation"] != IMPLEMENTATION
        or frozen["adapters"] is not None
        or frozen["shutdown_reserve_seconds"] != 20
        or frozen["cpu_prepare_timeout_seconds"] != 900
        or frozen["server_ready_timeout_seconds"] != 180
        or frozen["automatic_retry"] is not False
        or frozen["raw_records_uploaded"] is not False
        or not REQUIRED_INPUTS.issubset(frozen["files"])
        or frozen["protocol_text"] != PROTOCOL.read_text()
        or frozen["protocol_sha256"] != stream_sha(PROTOCOL)
    ):
        raise ValueError("Invalid frozen calibration plan")
    for path, expected in frozen["files"].items():
        if stream_sha(path) != expected:
            raise ValueError("Frozen calibration input changed: " + path)
    if (
        stream_sha(Path(frozen["config"]["identity_root"]) / "freeze.json")
        != IDENTITY_SHA
    ):
        raise ValueError("Goal identity profile changed")
    if stream_sha(PARENT / "freeze.json") != PARENT_SHA:
        raise ValueError("Parent identity changed")
    return frozen


def require_parent_completed():
    ledger = read(PARENT / "execution.json")
    if ledger.get("status") != "completed" or ledger.get("open"):
        raise ValueError("Stage 4 has not completed; never share its model servers")
    cases = ledger.get("cases", {})
    if len(cases) != 100 or any(c.get("status") != "completed" for c in cases.values()):
        raise ValueError("Stage 4 model inventory is incomplete")
    report = read("output/variance-diagnostic-v1/stage4/summary.json")
    if (
        not report.get("complete")
        or report.get("cases") != 100
        or report.get("actual_evaluation_episodes") != 6400
        or report.get("freeze_sha256") != PARENT_SHA
    ):
        raise ValueError("The audited complete parent report is not available")
    return dict(
        parent_report_sha256=stream_sha(
            "output/variance-diagnostic-v1/stage4/summary.json"
        )
    )


def subprocess_environment(gpu=None):
    env = dict(
        os.environ,
        PYTHONPATH=str(Path("src").absolute()),
        CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    # No inherited teacher or unrelated cloud credentials in local subprocesses.
    for key in list(env):
        if key.startswith(("LH_TEACHER_", "WANDB_", "OPENAI_")):
            env.pop(key)
    for key in ("NO_PROXY", "no_proxy"):
        env[key] = ",".join(filter(None, [env.get(key, ""), "127.0.0.1,localhost,::1"]))
    if gpu is None:
        env["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
        env["PATH"] = env["JAVA_HOME"] + "/bin:" + env.get("PATH", "")
    return env


def validate_health(health, frozen, alias, server_pid):
    manifest = Path(frozen["model_directory"]) / "download-manifest.json"
    expected = dict(
        model=alias,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        revision=REVISION,
        adapters=None,
        manifest_sha256=frozen["files"][str(manifest)],
        weights_sha256=frozen["weights_sha256"],
        ready=True,
        implementation=IMPLEMENTATION,
        pid=server_pid,
        context_tokens=8192,
        max_output_tokens=100,
        stop=["\n"],
        dtype="bfloat16",
        attention="sdpa",
        runtime=frozen["model_runtime"],
    )
    if any(health.get(k) != value for k, value in expected.items()):
        raise ValueError("Dedicated base server identity or execution profile differs")


def recover_aggregate(root, frozen):
    """Only this result, not per-episode rows/journals, may be uploaded."""
    root = Path(root)
    groups = {}
    for mode in frozen["config"]["prompt_modes"]:
        planned = [c for c in frozen["cases"] if c["mode"] == mode]
        results = []
        journals = []
        for case in planned:
            directory = root / "local" / case["name"]
            if (directory / "result.json").exists():
                result = read(directory / "result.json")
                if result["case"] != case or result["freeze_sha256"] != stream_sha(
                    root / "freeze.json"
                ):
                    raise ValueError("Mismatched episode result")
                journal = directory / "requests.jsonl"
                if not journal.exists():
                    raise ValueError("Completed episode is missing request accounting")
                usage = summarize_requests(journal)
                value = result["result"]
                if (
                    not usage["complete_usage"]
                    or value["model_calls"] != usage["attempted_requests"]
                    or value["input_tokens"] != usage["known_input_tokens"]
                    or value["output_tokens"] != usage["known_output_tokens"]
                ):
                    raise ValueError(
                        "Episode result differs from durable request usage"
                    )
                results.append(result["result"])
            if (directory / "requests.jsonl").exists():
                journals.append(summarize_requests(directory / "requests.jsonl"))
        groups[mode] = dict(
            planned_episodes=len(planned),
            completed_episodes=len(results),
            episodes_with_request_journal=len(journals),
            mean_success_completed_only=(
                sum(r["success"] for r in results) / len(results)
            )
            if results
            else None,
            mean_score_completed_only=(sum(r["score"] for r in results) / len(results))
            if results
            else None,
            decision_budget_episodes=sum(
                r["status"] == "decision_budget" for r in results
            ),
            **{
                key: sum(j[key] for j in journals)
                for key in (
                    "attempted_requests",
                    "response_records",
                    "error_records",
                    "pending_requests",
                    "unknown_usage_requests",
                    "known_input_tokens",
                    "known_output_tokens",
                )
            },
            complete_usage=all(j["complete_usage"] for j in journals),
            **{
                key: sum(r[key] for r in results)
                for key in (
                    "physical_actions",
                    "thought_decisions",
                    "invalid_format_decisions",
                    "invalid_control_decisions",
                )
            },
        )
    complete = sum(g["completed_episodes"] for g in groups.values()) == 64
    return dict(
        complete=complete,
        calibration_only=True,
        freeze_sha256=stream_sha(root / "freeze.json"),
        groups=groups,
        completed_episodes=sum(g["completed_episodes"] for g in groups.values()),
        planned_episodes=64,
        teacher_calls=0,
        eval_episodes=0,
        official_test_evaluations=0,
        raw_records_uploaded=False,
        note="Train-only public-prompt chat/native port; partial completed-only scores are not a paired baseline comparison or paper benchmark result.",
    )
