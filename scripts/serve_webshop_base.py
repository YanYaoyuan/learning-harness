#!/usr/bin/env python3
"""No adapters, local weights only, serial greedy generation with newline stop."""

import argparse
import json

from learning_harness.webshop_serving import BaseEngine, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("Use an unprivileged local port")
    engine = BaseEngine(args.model, args.revision, args.alias)
    print(json.dumps(engine.health), flush=True)
    serve(engine, args.port)


if __name__ == "__main__":
    main()
