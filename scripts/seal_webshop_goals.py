#!/usr/bin/env python3
"""Freeze then seal full-catalog goal identities without resetting an episode."""

import argparse
from collections import Counter
import importlib.metadata
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

from learning_harness.bounded_queue import save
from learning_harness.common import digest, file_sha

BASE = Path("output/webshop-preparation/20261113")
SOURCE = Path("third_party/WebShop-2022-reference")
PROTOCOL = Path("docs/WEBSHOP_GOAL_IDENTITY_PROTOCOL_2026-11-13.zh-CN.md")


def runtime():
    return dict(
        python=sys.version,
        packages={
            k: importlib.metadata.version(k)
            for k in ["numpy", "torch", "spacy", "pyserini", "ijson", "gym"]
        },
    )


def freeze(out, destination):
    if (out / "freeze.json").exists() or destination.exists():
        raise ValueError("Keep an existing identity freeze or destination")
    catalog = json.loads((BASE / "full-catalog/download.json").read_text())
    index = json.loads((BASE / "full-index/verification.json").read_text())
    preview = json.loads(
        Path("output/webshop-preparation/20261112/smoke.json").read_text()
    )
    reference = json.loads(
        Path(
            "output/webshop-preparation/20261112/original-reference/reference.json"
        ).read_text()
    )
    if (
        index["status"] != "full_index_verified"
        or reference["status"] != "reference_tests_passed"
    ):
        raise ValueError("Verified full index and original reward reference required")
    data = {r["name"]: r["path"] for r in catalog["files"]}
    human = next(r for r in preview["data"] if r["name"] == "items_human_ins.json")
    if file_sha(human["path"]) != human["sha256"]:
        raise ValueError("Human annotation source changed")
    paths = [
        Path(__file__).absolute(),
        PROTOCOL,
        BASE / "full-catalog/download.json",
        BASE / "full-index/verification.json",
        Path("output/webshop-preparation/20261112/original-reference/reference.json"),
        Path("third_party/WebShop/baseline_models/env.py"),
    ]
    paths += [
        SOURCE / p
        for p in [
            "web_agent_site/engine/engine.py",
            "web_agent_site/engine/goal.py",
            "web_agent_site/envs/web_agent_text_env.py",
            "web_agent_site/utils.py",
        ]
    ]
    paths += [Path(p) for p in data.values()] + [Path(human["path"])]
    paths += [Path(index["index"]) / p for p in index["index_files"]]
    out.mkdir(parents=True, exist_ok=True)
    value = dict(
        protocol="webshop-goal-identity-v1",
        frozen_unix=time.time(),
        seed_before_load=11,
        upstream_shuffle_seed=233,
        test_range=[0, 500],
        eval_range=[500, 1500],
        train_start=1500,
        data=data,
        human_annotations=human["path"],
        index=index["index"],
        nlp_pythonpath=reference["nlp_pythonpath"],
        destination=str(destination),
        runtime=runtime(),
        protocol_text=PROTOCOL.read_text(),
        files={str(p): file_sha(p) for p in paths},
        model_calls=0,
        official_test_evaluations=0,
    )
    with (out / "freeze.json").open("x") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
    return value


def run(out):
    frozen = json.loads((out / "freeze.json").read_text())
    if runtime() != frozen["runtime"] or frozen["seed_before_load"] != 11:
        raise ValueError("Frozen identity runtime or seed changed")
    for path, expected in frozen["files"].items():
        if file_sha(path) != expected:
            raise ValueError("Identity input changed: " + path)
    receipt = out / "sealed.json"
    destination = Path(frozen["destination"])
    if receipt.exists() or destination.exists():
        raise ValueError("Keep the previous identity attempt")
    destination.mkdir(parents=True, exist_ok=False)
    state = dict(
        status="loading_full_catalog",
        pid=os.getpid(),
        started_unix=time.time(),
        freeze_sha256=file_sha(out / "freeze.json"),
        model_calls=0,
        gpu_used=False,
        official_test_evaluations=0,
        full_benchmark_ready=False,
    )
    save(receipt, state)
    os.nice(10)
    try:
        sys.path[:0] = [str(SOURCE.absolute()), frozen["nlp_pythonpath"]]
        import numpy as np
        import torch
        import web_agent_site.engine.engine as engine
        from web_agent_site.engine.goal import nlp
        from pyserini.search.lucene import LuceneSearcher

        module = importlib.import_module("web_agent_site.envs.web_agent_text_env")
        if torch.cuda.is_available() or nlp.meta["name"] != "core_web_lg":
            raise ValueError("Wrong CPU/reward reference")
        engine.DEFAULT_ATTR_PATH = frozen["data"]["items_ins_v2.json"]
        engine.HUMAN_ATTR_PATH = frozen["human_annotations"]

        def search_engine(num_products=None):
            if num_products is not None:
                raise ValueError("Only the frozen full catalog is permitted")
            return LuceneSearcher(frozen["index"])

        module.init_search_engine = search_engine
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        server = module.SimServer(
            "http://127.0.0.1:3000",
            frozen["data"]["items_shuffle.json"],
            human_goals=True,
        )
        if (
            len(server.all_products) != 1181430
            or len(server.goals) <= 1500
            or server.user_sessions
        ):
            raise ValueError(
                "Wrong full catalog, goal count or unexpected episode reset"
            )
        counts = Counter()
        full_hashes = {s: set() for s in ["test", "eval", "train"]}
        visible_hashes = {s: set() for s in full_hashes}
        products = {s: set() for s in full_hashes}
        manifest = destination / "identities.jsonl"
        with manifest.open("x") as handle:
            for i, goal in enumerate(server.goals):
                split = "test" if i < 500 else "eval" if i < 1500 else "train"
                full, visible = digest(goal), digest(goal["instruction_text"])
                full_hashes[split].add(full)
                visible_hashes[split].add(visible)
                products[split].add(goal["asin"])
                counts[split] += 1
                handle.write(
                    json.dumps(
                        dict(
                            global_index=i,
                            split=split,
                            goal_sha256=full,
                            visible_instruction_sha256=visible,
                        )
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        if (
            counts["test"] != 500
            or counts["eval"] != 1000
            or sum(counts.values()) != len(server.goals)
        ):
            raise ValueError("Standard index-based split is incomplete")
        overlap = {}
        for first, second in [("train", "test"), ("train", "eval"), ("eval", "test")]:
            overlap[first + "_" + second] = dict(
                full_goal_fingerprints=len(full_hashes[first] & full_hashes[second]),
                visible_instruction_fingerprints=len(
                    visible_hashes[first] & visible_hashes[second]
                ),
                products=len(products[first] & products[second]),
            )
        state.update(
            status="goal_identities_sealed",
            finished_unix=time.time(),
            products=len(server.all_products),
            goals=len(server.goals),
            split_counts=dict(counts),
            index_ranges_disjoint=True,
            content_overlap_counts=overlap,
            identity_manifest=str(manifest),
            identity_manifest_sha256=file_sha(manifest),
            human_goal_text_written=False,
            episode_resets=0,
            user_sessions=0,
            remaining=[
                "Permitted-pool full environment factory and public interface integration",
                "Formal model/collection/training/evaluation protocol",
            ],
        )
        save(receipt, state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(
            status="failed", error_type=type(error).__name__, stopped_unix=time.time()
        )
        save(receipt, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        if args.destination is None:
            parser.error("--destination is required when freezing")
        value = freeze(args.out, args.destination)
        print(
            json.dumps(
                dict(
                    status="identity_protocol_frozen",
                    files=len(value["files"]),
                    model_calls=0,
                )
            ),
            flush=True,
        )
    else:
        run(args.out)


if __name__ == "__main__":
    main()
