"""Validate explicitly recorded idle desktop contexts without stopping other apps."""

import json
import os
from pathlib import Path
import urllib.request


def process_identity(pid):
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [
        v.decode(errors="replace")
        for v in (root / "cmdline").read_bytes().split(b"\0")
        if v
    ]
    return dict(
        pid=pid,
        start_ticks=int(stat[19]),
        uid=root.stat().st_uid,
        executable=str((root / "exe").resolve()),
        scripts=[v for v in argv if v.endswith(".py")],
    ), argv


def validate_idle_context(record, used_mib):
    identity, argv = process_identity(record["pid"])
    if (
        any(identity[k] != record[k] for k in identity)
        or identity["uid"] != os.getuid()
    ):
        raise ValueError("Recorded background PID identity changed")
    cap = record["max_memory_mib"]
    if not 0 < cap <= 1024 or not 0 <= used_mib <= cap:
        raise ValueError("Background GPU residency exceeds the frozen allowance")
    role = record["role"]
    result = dict(pid=record["pid"], role=role, used_memory_mib=used_mib)
    if role == "idle_comfyui":
        if len(identity["scripts"]) != 1 or not identity["scripts"][0].endswith(
            "/ComfyUI/main.py"
        ):
            raise ValueError("Only the recorded ComfyUI server may use this allowance")
        port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8188
        if port != record["queue_port"] or not 1 <= port <= 65535:
            raise ValueError("Recorded ComfyUI queue port changed")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/queue", timeout=5) as response:
            queue = json.load(response)
        if (
            not isinstance(queue["queue_running"], list)
            or not isinstance(queue["queue_pending"], list)
            or queue["queue_running"]
            or queue["queue_pending"]
        ):
            raise ValueError("ComfyUI has an active or queued job; leave it untouched")
        result.update(queue_running=0, queue_pending=0)
    elif role == "desktop_terminal":
        if Path(identity["executable"]).name != "omniterm-app":
            raise ValueError("Unrecognized desktop context")
    else:
        raise ValueError("Unknown background-context role")
    return result
