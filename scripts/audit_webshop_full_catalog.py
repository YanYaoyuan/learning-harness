#!/usr/bin/env python3
"""Stream product/schema counts without constructing or opening human goals."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import ijson


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Keep an existing schema audit; use a new output path")
    download = json.loads(args.download.read_text())
    if download["status"] != "catalog_files_checksum_verified":
        raise ValueError("Wait for the original acquisition to finish")
    os.nice(10)
    files = {r["name"]: r for r in download["files"]}
    began = time.monotonic()
    for r in files.values():
        if sha(Path(r["path"])) != r["actual_sha256"]:
            raise ValueError("Downloaded product file has changed")
    required = [
        "asin",
        "name",
        "full_description",
        "small_description",
        "pricing",
        "images",
        "customization_options",
        "query",
        "category",
        "product_category",
    ]
    counts, missing, incompatible = Counter(), Counter(), Counter()
    all_asins, usable_asins = set(), set()
    with Path(files["items_shuffle.json"]["path"]).open("rb") as handle:
        for product in ijson.items(handle, "item"):
            if time.monotonic() - began > 300:
                raise TimeoutError("Five-minute CPU catalog audit cap reached")
            counts["raw_products"] += 1
            if not isinstance(product, dict):
                raise ValueError("Product array contains a non-object")
            missing.update(k for k in required if k not in product)
            asin = product.get("asin")
            if not isinstance(asin, str):
                incompatible["asin_not_string"] += 1
                continue
            if asin in all_asins:
                counts["duplicate_asin_rows"] += 1
            all_asins.add(asin)
            if asin == "nan" or len(asin) > 10:
                counts["upstream_filtered_asin_rows"] += 1
                continue
            if asin in usable_asins:
                counts["upstream_duplicate_rows"] += 1
                continue
            usable_asins.add(asin)
            counts["upstream_usable_products"] += 1
            for key in [
                "name",
                "full_description",
                "query",
                "category",
                "product_category",
            ]:
                if not isinstance(product.get(key), str):
                    incompatible[key + "_not_string"] += 1
            images = product.get("images")
            if not isinstance(images, list) or not images:
                incompatible["missing_first_image"] += 1
            if counts["raw_products"] % 200000 == 0:
                print(
                    json.dumps(
                        dict(
                            raw_products=counts["raw_products"],
                            cpu_elapsed_seconds=time.monotonic() - began,
                        )
                    ),
                    flush=True,
                )
    attrs, attrs_missing, products_with_attributes = 0, 0, 0
    attribute_keys = set()
    with Path(files["items_ins_v2.json"]["path"]).open("rb") as handle:
        for asin, value in ijson.kvitems(handle, ""):
            attrs += 1
            if asin in attribute_keys:
                raise ValueError("Duplicate top-level attribute key")
            attribute_keys.add(asin)
            if not isinstance(value, dict):
                raise ValueError("Attribute mapping contains a non-object")
            if "attributes" not in value:
                attrs_missing += 1
            elif asin in usable_asins:
                products_with_attributes += 1
    if counts["raw_products"] < 1000000:
        raise ValueError("Expected a full catalog, not a smaller preview")
    result = dict(
        status="full_catalog_schema_audited",
        checked_unix=time.time(),
        acquisition_sha256=sha(args.download),
        file_sha256={k: r["actual_sha256"] for k, r in files.items()},
        parser=dict(name="ijson", version=ijson.__version__, backend=ijson.backend),
        counts=dict(counts),
        unique_raw_asins=len(all_asins),
        required_field_missing_counts=dict(missing),
        possible_upstream_incompatibilities=dict(incompatible),
        attribute_records=attrs,
        attribute_records_without_attributes=attrs_missing,
        usable_products_with_attribute_record=products_with_attributes,
        usable_products_without_attribute_record=len(usable_asins)
        - products_with_attributes,
        cpu_elapsed_seconds=time.monotonic() - began,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        human_annotation_file_opened=False,
        goal_mapping_built=False,
        official_test_evaluations=0,
        model_calls=0,
        gpu_used=False,
        full_benchmark_ready=False,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
