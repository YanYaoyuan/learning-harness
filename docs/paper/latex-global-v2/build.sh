#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
paper_build_dir="${PAPER_BUILD_DIR:-build}"
mkdir -p "$paper_build_dir"
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$paper_build_dir" main.tex
elif [ -n "${TECTONIC_BIN:-}" ]; then
  "$TECTONIC_BIN" --keep-logs --keep-intermediates --outdir "$paper_build_dir" main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --keep-intermediates --outdir "$paper_build_dir" main.tex
else
  echo 'Install TeX Live with latexmk, or set TECTONIC_BIN; Overleaf can compile main.tex directly.' >&2
  exit 1
fi
