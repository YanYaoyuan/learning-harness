#!/usr/bin/env python3
"""Restore archived JSONL traces after checksum and conflict checks."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='.')
    parser.add_argument('--study',help='Optional study directory name, for example baseline-benchmark-v1')
    args=parser.parse_args();root=Path(args.root).resolve()
    manifest=json.loads((root/'SNAPSHOT_MANIFEST.json').read_text());count=0
    for entry in manifest['files']:
        if not entry['path'].endswith('.jsonl.gz'):continue
        if args.study and args.study not in Path(entry['source_path']).parts:continue
        source=(root/entry['path']).resolve();destination=(root/entry['source_path']).resolve()
        if not source.is_relative_to(root) or not destination.is_relative_to(root):raise ValueError('Unsafe snapshot path')
        payload=source.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=entry['sha256']:raise ValueError('Compressed trace checksum mismatch')
        decoded=gzip.decompress(payload)
        if hashlib.sha256(decoded).hexdigest()!=entry['source_snapshot_sha256']:raise ValueError('Trace checksum mismatch')
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=entry['source_snapshot_sha256']:
                raise ValueError('Refusing to overwrite different data: '+str(destination))
        else:destination.write_bytes(decoded)
        count+=1
    print(f'Verified/restored {count} trace files.')


if __name__=='__main__':main()
