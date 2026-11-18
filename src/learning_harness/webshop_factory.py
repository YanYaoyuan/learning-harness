"""Construct split-restricted original WebShop servers before any auto-reset."""

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import uuid

from .common import digest, read_jsonl
from .webshop_public import PublicWebShop


def stream_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


class GoalPoolGate:
    def __init__(self, records, split):
        if split not in ("train", "eval", "test"):
            raise ValueError("Unknown split")
        for index, row in enumerate(records):
            expected = "test" if index < 500 else "eval" if index < 1500 else "train"
            if row["global_index"] != index or row["split"] != expected:
                raise ValueError("Sealed identity order or standard split changed")
        self.records = records
        self.split = split
        self.selected = [r["global_index"] for r in records if r["split"] == split]
        if not self.selected:
            raise ValueError("Requested pool is empty")
        self._locals = {
            global_index: local for local, global_index in enumerate(self.selected)
        }
        self.checked = 0

    def __call__(self, index, goal):
        if (
            type(index) is not int
            or index != self.checked
            or index >= len(self.records)
        ):
            raise ValueError("Upstream goal traversal differs from the sealed profile")
        row = self.records[index]
        if (
            digest(goal) != row["goal_sha256"]
            or digest(goal["instruction_text"]) != row["visible_instruction_sha256"]
        ):
            raise ValueError("Generated goal differs from its sealed identity")
        self.checked += 1
        return index in self._locals

    def finish(self):
        if self.checked != len(self.records):
            raise ValueError("Not every original goal was checked before filtering")

    def local_index(self, global_index):
        self.finish()
        if type(global_index) is not int or global_index not in self._locals:
            raise ValueError("Requested global goal is outside the permitted split")
        return self._locals[global_index]


class BoundEpisode:
    def __init__(self, environment, server, prefix, local_index, expected_hash):
        self._environment = environment
        self._server = server
        self._prefix = prefix
        self._local_index = local_index
        self._expected_hash = expected_hash
        self._public = PublicWebShop(environment, [local_index])

    def reset(self):
        view = self._public.reset(self._local_index)
        goal = self._server.user_sessions[self._environment.session]["goal"]
        if digest(goal) != self._expected_hash:
            raise ValueError("Explicit reset selected a different global goal")
        return view

    def step(self, action):
        return self._public.step(action)

    def close(self):
        self._environment.close()
        for session in list(self._server.user_sessions):
            if session.startswith(self._prefix):
                del self._server.user_sessions[session]


class SplitEnvironment:
    def __init__(self, server, environment_class, gate, metadata):
        gate.finish()
        if len(server.goals) != len(gate.selected) or server.user_sessions:
            raise ValueError("Server is not restricted before environment construction")
        for global_index, goal in zip(gate.selected, server.goals):
            if digest(goal) != gate.records[global_index]["goal_sha256"]:
                raise ValueError("Filtered pool order differs from the sealed mapping")
        self._server = server
        self._environment_class = environment_class
        self._gate = gate
        self.metadata = dict(
            metadata,
            split=gate.split,
            permitted_goals=len(gate.selected),
            verified_global_goals=gate.checked,
            server_sessions_before_first_environment=0,
        )

    def episode(self, global_index):
        local = self._gate.local_index(global_index)
        prefix = "lh_" + uuid.uuid4().hex + "_"
        environment = self._environment_class(
            observation_mode="text",
            server=self._server,
            get_image=False,
            session_prefix=prefix,
        )
        return BoundEpisode(
            environment,
            self._server,
            prefix,
            local,
            self._gate.records[global_index]["goal_sha256"],
        )


def build_full_environment(identity_root, split):
    """Backend only. Test episodes require a separately frozen study protocol.

    This function creates a server, not an episode. The caller explicitly opens
    only the split authorized by its study; preparation uses train exclusively.
    """
    root = Path(identity_root)
    frozen = json.loads((root / "freeze.json").read_text())
    sealed = json.loads((root / "sealed.json").read_text())
    if (
        sealed["status"] != "goal_identities_sealed"
        or stream_sha(root / "freeze.json") != sealed["freeze_sha256"]
    ):
        raise ValueError("Missing or mismatched sealed profile")
    current = dict(
        python=sys.version,
        packages={
            k: importlib.metadata.version(k) for k in frozen["runtime"]["packages"]
        },
    )
    if (
        current != frozen["runtime"]
        or frozen["seed_before_load"] != 11
        or frozen["upstream_shuffle_seed"] != 233
    ):
        raise ValueError("Runtime or random-number profile changed")
    for path, expected in frozen["files"].items():
        if stream_sha(path) != expected:
            raise ValueError("Frozen profile input changed: " + path)
    manifest = Path(sealed["identity_manifest"])
    if stream_sha(manifest) != sealed["identity_manifest_sha256"]:
        raise ValueError("Sealed per-goal manifest changed")
    records = read_jsonl(manifest)
    if len(records) != sealed["goals"]:
        raise ValueError("Sealed goal count changed")
    gate = GoalPoolGate(records, split)
    source = Path("third_party/WebShop-2022-reference").absolute()
    sys.path[:0] = [str(source), frozen["nlp_pythonpath"]]
    import numpy as np
    import torch
    import web_agent_site.engine.engine as engine
    from web_agent_site.engine.goal import nlp
    from pyserini.search.lucene import LuceneSearcher

    module = importlib.import_module("web_agent_site.envs.web_agent_text_env")
    if (
        not Path(module.__file__).resolve().is_relative_to(source)
        or torch.cuda.is_available()
        or nlp.meta["name"] != "core_web_lg"
    ):
        raise ValueError("Use an isolated process with the frozen original CPU profile")
    engine.DEFAULT_ATTR_PATH = frozen["data"]["items_ins_v2.json"]
    engine.HUMAN_ATTR_PATH = frozen["human_annotations"]

    def search_engine(num_products=None):
        if num_products is not None:
            raise ValueError("Only the complete catalog is permitted")
        return LuceneSearcher(frozen["index"])

    module.init_search_engine = search_engine
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    server = module.SimServer(
        "http://127.0.0.1:3000",
        frozen["data"]["items_shuffle.json"],
        human_goals=True,
        filter_goals=gate,
    )
    if len(server.all_products) != sealed["products"]:
        raise ValueError("Full catalog size changed")
    return SplitEnvironment(
        server,
        module.WebAgentTextEnv,
        gate,
        dict(
            identity_freeze_sha256=sealed["freeze_sha256"],
            identity_manifest_sha256=sealed["identity_manifest_sha256"],
            products=len(server.all_products),
            indexed_documents=server.search_engine.num_docs,
            factory_sha256=stream_sha(__file__),
        ),
    )
