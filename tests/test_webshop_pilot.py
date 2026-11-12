import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from learning_harness.webshop_factory import stream_sha
import learning_harness.webshop_pilot as pilot
from learning_harness.webshop_requests import RequestJournal
from learning_harness.webshop_serving import IMPLEMENTATION


CONFIG = Path("configs/webshop-development-pilot-v1.json")


def test_all_cases_are_fixed_train_targets_paired_and_order_balanced():
    cases = pilot.episode_plan(pilot.read(CONFIG))
    assert len(cases) == 64 and len({c["name"] for c in cases}) == 64
    assert {c["global_goal_index"] for c in cases} == set(range(1500, 1532))
    for i in range(32):
        a, b = cases[i * 2 : i * 2 + 2]
        assert a["global_goal_index"] == b["global_goal_index"]
        assert {a["mode"], b["mode"]} == {"prompt1", "prompt1_actonly"}
        assert a["mode"] == ("prompt1" if i % 2 == 0 else "prompt1_actonly")


@pytest.mark.parametrize(
    "key,value",
    [
        ("global_goal_indices", [0]),
        ("split", "test"),
        ("student_adapters", "weights"),
        ("max_executed_decisions", 15),
        ("teacher_calls", 1),
        ("existing_model_servers_may_be_queried", True),
        ("rollout_seed", True),
    ],
)
def test_scope_cannot_silently_change(key, value):
    config = pilot.read(CONFIG)
    config[key] = value
    with pytest.raises(ValueError):
        pilot.validate_config(config)


def test_child_environments_keep_localhost_and_hide_cpu_cuda(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "existing.internal")
    monkeypatch.setenv("LH_TEACHER_API_KEY", "dummy")
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    cpu = pilot.subprocess_environment()
    gpu = pilot.subprocess_environment(1)
    assert cpu["CUDA_VISIBLE_DEVICES"] == "" and gpu["CUDA_VISIBLE_DEVICES"] == "1"
    assert cpu["NO_PROXY"] == "existing.internal,127.0.0.1,localhost,::1"
    assert "LH_TEACHER_API_KEY" not in cpu and "WANDB_API_KEY" not in gpu
    assert cpu["JAVA_HOME"].endswith("java-11-openjdk-amd64")
    assert gpu["HF_HUB_OFFLINE"] == "1"


def test_dedicated_health_checks_process_weights_and_adapter_identity():
    manifest = "data/models/Qwen3-4B-Instruct-2507/download-manifest.json"
    frozen = dict(
        model_directory=str(Path(manifest).parent),
        files={manifest: "manifest-hash"},
        weights_sha256="weight-hash",
        model_runtime={"test": "runtime"},
    )
    health = dict(
        model="alias",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        revision=pilot.REVISION,
        adapters=None,
        manifest_sha256="manifest-hash",
        weights_sha256="weight-hash",
        ready=True,
        implementation=IMPLEMENTATION,
        pid=123,
        context_tokens=8192,
        max_output_tokens=100,
        stop=["\n"],
        dtype="bfloat16",
        attention="sdpa",
        runtime={"test": "runtime"},
    )
    pilot.validate_health(health, frozen, "alias", 123)
    for changed in [
        dict(pid=124),
        dict(adapters="some-adapter"),
        dict(weights_sha256="wrong"),
        dict(stop=[]),
    ]:
        with pytest.raises(ValueError):
            pilot.validate_health(dict(health, **changed), frozen, "alias", 123)


def test_running_parent_blocks_calibration_before_any_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "PARENT", tmp_path)
    (tmp_path / "execution.json").write_text(json.dumps(dict(status="running")))
    with pytest.raises(ValueError, match="has not completed"):
        pilot.require_parent_completed()


def test_partial_aggregate_retains_failed_request_and_no_task_identifiers(tmp_path):
    config = pilot.read(CONFIG)
    frozen = dict(config=config, cases=pilot.episode_plan(config))
    (tmp_path / "freeze.json").write_text(json.dumps(frozen))
    case = frozen["cases"][0]
    directory = tmp_path / "local" / case["name"]
    with RequestJournal(directory / "requests.jsonl") as journal:
        journal(
            dict(kind="attempt", turn=0, messages=[dict(content="PRIVATE-GOAL-TEXT")])
        )
        journal(dict(kind="error", turn=0, error_type="TimeoutError"))
    result = pilot.recover_aggregate(tmp_path, frozen)
    mode = result["groups"][case["mode"]]
    assert not result["complete"] and result["completed_episodes"] == 0
    assert mode["unknown_usage_requests"] == mode["attempted_requests"] == 1
    assert not mode["complete_usage"] and mode["mean_success_completed_only"] is None
    serialized = json.dumps(result)
    assert "PRIVATE-GOAL-TEXT" not in serialized and "1500" not in serialized


def test_successful_result_must_match_durable_token_usage(tmp_path):
    config = pilot.read(CONFIG)
    frozen = dict(config=config, cases=pilot.episode_plan(config))
    (tmp_path / "freeze.json").write_text(json.dumps(frozen))
    case = frozen["cases"][0]
    directory = tmp_path / "local" / case["name"]
    with RequestJournal(directory / "requests.jsonl") as journal:
        journal(dict(kind="attempt", turn=0, messages=[]))
        journal(
            dict(
                kind="response", turn=0, response=dict(input_tokens=5, output_tokens=3)
            )
        )
    result = dict(
        status="decision_budget",
        success=False,
        score=0.0,
        model_calls=1,
        input_tokens=5,
        output_tokens=3,
        physical_actions=0,
        thought_decisions=1,
        invalid_format_decisions=0,
        invalid_control_decisions=0,
    )
    record = dict(
        case=case, result=result, freeze_sha256=stream_sha(tmp_path / "freeze.json")
    )
    (directory / "result.json").write_text(json.dumps(record))
    assert pilot.recover_aggregate(tmp_path, frozen)["completed_episodes"] == 1
    record["result"]["input_tokens"] = 0
    (directory / "result.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="durable request"):
        pilot.recover_aggregate(tmp_path, frozen)


def test_controller_cleanup_reaps_owned_process_group():
    path = Path("scripts/run_webshop_pilot.py")
    spec = importlib.util.spec_from_file_location("webshop_pilot_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        module.stop_children([child])
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            os.killpg(child.pid, 9)
            child.wait(timeout=5)


@pytest.mark.parametrize("outcome", ["completed", "failed", "stopped_at_cap"])
def test_controller_real_subprocess_lifecycle_without_gpu(
    tmp_path, monkeypatch, outcome
):
    """Real child groups/HTTP/ledger, explicit stand-ins instead of CUDA/data."""
    monkeypatch.syspath_prepend(str(Path("scripts").absolute()))
    import action_control_report
    import run_action_controls

    spec = importlib.util.spec_from_file_location(
        "pilot_lifecycle_test", "scripts/run_webshop_pilot.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = pilot.read(CONFIG)
    manifest = "data/models/Qwen3-4B-Instruct-2507/download-manifest.json"
    frozen = dict(
        config=config,
        cases=pilot.episode_plan(config),
        gpu=0,
        port=0,
        files={manifest: "manifest-hash"},
        weights_sha256="weights-hash",
        model_runtime={"test": "runtime"},
        model_directory=str(Path(manifest).parent),
        allowed_background_contexts=[],
        model_python=sys.executable,
        environment_python=sys.executable,
        cap_gpu_seconds=2.5,
        shutdown_reserve_seconds=0.8,
        cpu_prepare_timeout_seconds=5,
        server_ready_timeout_seconds=2,
    )
    # Reserve an ephemeral local port; the dedicated child claims it on launch.
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        frozen["port"] = sock.getsockname()[1]
    (tmp_path / "freeze.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(runner, "validate_frozen", lambda root: frozen)
    monkeypatch.setattr(runner, "require_parent_completed", lambda: {})
    monkeypatch.setattr(action_control_report, "load_study", lambda root: None)
    monkeypatch.setattr(run_action_controls, "gpu_preflight", lambda *args: [])
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(key, "127.0.0.1,localhost,::1")
    stub = tmp_path / "explicit_no_gpu_standin.py"
    stub.write_text("""
import json,os,pathlib,sys,time
from learning_harness.bounded_queue import save
from learning_harness.webshop_factory import stream_sha
from learning_harness.webshop_pilot import REVISION
from learning_harness.webshop_serving import IMPLEMENTATION,serve
root=pathlib.Path(sys.argv[1]);role=sys.argv[2];outcome=sys.argv[3]
frozen=json.loads((root/'freeze.json').read_text())
sha=stream_sha(root/'freeze.json')
if role=='server':
 class Engine:
  health=dict(model='webshop-base-'+sha[:12],base_model='Qwen/Qwen3-4B-Instruct-2507',revision=REVISION,
              adapters=None,manifest_sha256='manifest-hash',weights_sha256='weights-hash',ready=True,
              implementation=IMPLEMENTATION,pid=os.getpid(),context_tokens=8192,max_output_tokens=100,
              stop=['\\n'],dtype='bfloat16',attention='sdpa',runtime={'test':'runtime'})
 serve(Engine(),frozen['port'])
else:
 save(root/'backend-ready.json',dict(pid=os.getpid(),freeze_sha256=sha,
                                   backend=dict(split='train',permitted_goals=10587)))
 while not (root/'release.json').exists(): time.sleep(.02)
 if outcome!='completed':
  directory=root/'local'/frozen['cases'][0]['name'];directory.mkdir(parents=True)
  with (directory/'requests.jsonl').open('w') as handle:
   handle.write(json.dumps(dict(kind='attempt',turn=0,messages=[]))+'\\n');handle.flush();os.fsync(handle.fileno())
  if outcome=='failed': sys.exit(7)
  while True: time.sleep(.1)
 for case in frozen['cases']:
  directory=root/'local'/case['name'];directory.mkdir(parents=True)
  (directory/'requests.jsonl').write_text('')
  result=dict(status='decision_budget',success=False,score=0.,model_calls=0,input_tokens=0,output_tokens=0,
              physical_actions=0,thought_decisions=0,invalid_format_decisions=0,invalid_control_decisions=0)
  save(directory/'result.json',dict(case=case,result=result,freeze_sha256=sha))
""")
    actual_popen = subprocess.Popen

    def no_gpu_popen(argv, **kwargs):
        role = "worker" if "webshop_pilot_worker.py" in argv[1] else "server"
        return actual_popen(
            [sys.executable, str(stub), str(tmp_path), role, outcome], **kwargs
        )

    monkeypatch.setattr(runner.subprocess, "Popen", no_gpu_popen)
    if outcome == "completed":
        runner.run(tmp_path)
    else:
        with pytest.raises(RuntimeError, match="Calibration incomplete"):
            runner.run(tmp_path)
    execution = pilot.read(tmp_path / "execution.json")
    aggregate = pilot.read(tmp_path / "aggregate-summary.json")
    assert execution["status"] == outcome
    assert 0 < execution["gpu_seconds"] < 4
    assert all(code is not None for code in execution["children_returncodes"].values())
    assert aggregate["complete"] == (outcome == "completed")
    if outcome != "completed":
        first = config["prompt_modes"][0]
        assert aggregate["groups"][first]["unknown_usage_requests"] == 1
