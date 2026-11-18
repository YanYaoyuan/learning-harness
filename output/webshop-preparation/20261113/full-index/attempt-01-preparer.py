#!/usr/bin/env python3
"""Verify original preview retrieval, then build the full CPU-only search index."""

import argparse
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import ijson

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from learning_harness.webshop_index import search_document

QUERIES = (
    "shirt",
    "red shoes",
    "stainless steel",
    "coffee maker",
    "wireless",
    "blue cotton shirt",
    "nonexistentwordabcdefgh",
)
SOURCE = Path("third_party/WebShop-2022-reference")


def documents(source, folder):
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "documents.jsonl"
    seen, raw, filtered, duplicates = set(), 0, 0, 0
    with source.open("rb") as handle, path.open("x") as output:
        for product in ijson.items(handle, "item"):
            raw += 1
            asin = product["asin"]
            if asin == "nan" or len(asin) > 10:
                filtered += 1
                continue
            if asin in seen:
                duplicates += 1
                continue
            seen.add(asin)
            output.write(json.dumps(search_document(product)) + "\n")
            if raw % 200000 == 0:
                print(json.dumps(dict(phase="documents", raw_products=raw)), flush=True)
        output.flush()
        os.fsync(output.fileno())
    return dict(
        path=str(path),
        raw_products=raw,
        documents=len(seen),
        filtered=filtered,
        duplicates=duplicates,
        sha256=file_sha(path),
        bytes=path.stat().st_size,
    )


def compare_documents(candidate, reference):
    count = 0
    with candidate.open() as first, reference.open() as second:
        for a, b in zip_longest(first, second):
            if a is None or b is None:
                raise ValueError("Reference and minimal document counts differ")
            x, y = json.loads(a), json.loads(b)
            if x != {k: y[k] for k in ("id", "contents")}:
                raise ValueError("Document ID or contents differ from original preview")
            count += 1
    if count != 1000:
        raise ValueError("Expected all 1000 reference documents")
    return dict(
        documents=count,
        ids_and_contents_identical=True,
        reference_sha256=file_sha(reference),
        candidate_sha256=file_sha(candidate),
    )


def build(folder, index, seconds, log, state, receipt):
    argv = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(folder),
        "--index",
        str(index),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "1",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    with log.open("x") as handle:
        child = subprocess.Popen(
            argv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                JAVA_TOOL_OPTIONS="-Xmx4g",
                OMP_NUM_THREADS="1",
            ),
        )
        state["index_worker"] = dict(pid=child.pid, argv=argv, started_unix=time.time())
        save(receipt, state)
        try:
            code = child.wait(timeout=seconds)
        except BaseException:
            child.kill()
            child.wait()
            raise
    state.pop("index_worker")
    if code:
        raise RuntimeError("CPU Lucene index process failed; inspect its local log")


def search_check(path, reference=None, expected=1000):
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(path))
    if searcher.num_docs != expected:
        raise ValueError("Lucene document count differs from the audited catalog")
    original = LuceneSearcher(str(reference)) if reference else None
    results = []
    for query in QUERIES:
        hits = searcher.search(query, k=50)
        values = [(h.docid, h.score) for h in hits]
        if original:
            old = [(h.docid, h.score) for h in original.search(query, k=50)]
            if values != old:
                raise ValueError("Preview retrieval IDs, ordering or scores changed")
        for h in hits:
            raw = json.loads(searcher.doc(h.docid).raw())
            if set(raw) != {"id", "contents"} or raw["id"] != h.docid:
                raise ValueError("Raw search record exposes unexpected data")
        results.append(
            dict(
                query=query,
                hits=len(values),
                ordered_ids_and_scores_sha256=hashlib.sha256(
                    json.dumps(values).encode()
                ).hexdigest(),
            )
        )
    if original and original.num_docs != 1000:
        raise ValueError("Wrong preview reference index")
    return dict(
        documents=searcher.num_docs,
        queries=results,
        reference_rank_and_score_equality=bool(original),
        raw_fields=["id", "contents"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = args.out / "index.json"
    if receipt.exists() or args.destination.exists():
        raise ValueError("Preserve existing index attempt; inspect before restarting")
    preview = json.loads(
        Path("output/webshop-preparation/20261112/smoke.json").read_text()
    )
    acquisition = Path("output/webshop-preparation/20261113/full-catalog/download.json")
    audit = Path("output/webshop-preparation/20261113/full-catalog/schema-audit.json")
    full = json.loads(acquisition.read_text())
    schema = json.loads(audit.read_text())
    if full["status"] != "catalog_files_checksum_verified" or schema[
        "acquisition_sha256"
    ] != file_sha(acquisition):
        raise ValueError("Verified catalog and schema audit required")
    raw_full = next(r for r in full["files"] if r["name"] == "items_shuffle.json")
    small = next(r for r in preview["data"] if r["name"] == "items_shuffle_1000.json")
    if (
        file_sha(raw_full["path"]) != raw_full["actual_sha256"]
        or file_sha(small["path"]) != small["sha256"]
    ):
        raise ValueError("Catalog data changed")
    import shutil

    if (
        shutil.disk_usage(args.destination.parent).free
        < 2 * raw_full["downloaded_bytes"] + 8 * 1024**3
    ):
        raise ValueError(
            "Insufficient room for index preparation and future completed weights"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    args.destination.mkdir(parents=True, exist_ok=False)
    os.nice(10)
    state = dict(
        status="reference_check",
        started_unix=time.time(),
        pid=os.getpid(),
        source_revision="61e56dbba107ae13ff49e6058d7d2aad23326929",
        code_sha256={
            str(p): file_sha(p)
            for p in [
                Path(__file__),
                Path("src/learning_harness/webshop_index.py"),
                SOURCE / "search_engine/convert_product_file_format.py",
                SOURCE / "web_agent_site/engine/engine.py",
            ]
        },
        acquisition_sha256=file_sha(acquisition),
        schema_audit_sha256=file_sha(audit),
        catalog_sha256=raw_full["actual_sha256"],
        gpu_used=False,
        model_calls=0,
        human_annotation_file_opened=False,
        official_test_evaluations=0,
        full_benchmark_ready=False,
        original_environment_paths_modified=False,
    )
    save(receipt, state)
    try:
        small_docs = documents(
            Path(small["path"]), args.destination / "reference1000/resources"
        )
        reference = Path(
            "third_party/WebShop/search_engine/resources_1k/documents.jsonl"
        )
        state["reference_documents"] = compare_documents(
            Path(small_docs["path"]), reference
        )
        small_index = args.destination / "reference1000/index"
        build(
            Path(small_docs["path"]).parent,
            small_index,
            180,
            args.out / "reference-index.log",
            state,
            receipt,
        )
        state["reference_search"] = search_check(
            small_index, Path("third_party/WebShop/search_engine/indexes_1k")
        )
        state["status"] = "full_documents"
        save(receipt, state)
        full_docs = documents(
            Path(raw_full["path"]), args.destination / "full/resources"
        )
        if (
            full_docs["documents"] != schema["counts"]["upstream_usable_products"]
            or full_docs["raw_products"] != schema["counts"]["raw_products"]
        ):
            raise ValueError(
                "Converted document counts differ from original filtering audit"
            )
        state["full_documents"] = full_docs
        state["status"] = "full_index"
        save(receipt, state)
        index = args.destination / "full/index"
        build(
            Path(full_docs["path"]).parent,
            index,
            1800,
            args.out / "full-index.log",
            state,
            receipt,
        )
        state["full_search"] = search_check(index, expected=full_docs["documents"])
        state.update(
            status="full_index_verified",
            finished_unix=time.time(),
            index=str(index),
            index_files={
                str(p.relative_to(index)): dict(
                    bytes=p.stat().st_size, sha256=file_sha(p)
                )
                for p in sorted(index.rglob("*"))
                if p.is_file()
            },
            remaining=[
                "Goal identity and disjoint split protocol",
                "Public-only student interface",
                "Formal training and evaluation protocol",
            ],
        )
        save(receipt, state)
        print(
            json.dumps(
                {
                    k: state[k]
                    for k in [
                        "status",
                        "index",
                        "gpu_used",
                        "human_annotation_file_opened",
                        "full_benchmark_ready",
                    ]
                }
            ),
            flush=True,
        )
    except BaseException as error:
        state.update(
            status="failed", error_type=type(error).__name__, stopped_unix=time.time()
        )
        save(receipt, state)
        raise


if __name__ == "__main__":
    main()
