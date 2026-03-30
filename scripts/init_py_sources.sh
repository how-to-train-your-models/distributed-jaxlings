#!/usr/bin/env bash
# Bootstrap: convert existing chapters/*.ipynb → src/*.py
# Run this once to initialize the src/ directory from existing notebooks.
# After this, edit src/*.py and use generate_notebooks.sh to rebuild ipynb files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
SRC="$ROOT/src"
CHAPTERS="$ROOT/chapters"

JUPYTEXT="uv run jupytext"

mkdir -p "$SRC"

for nb in "$CHAPTERS"/chapter_*.ipynb; do
    name="$(basename "$nb" .ipynb)"
    out="$SRC/${name}.py"
    echo "Converting $name.ipynb → src/${name}.py"
    $JUPYTEXT --to py:percent --output "$out" "$nb"
done

echo ""
echo "Done. Edit files in src/ and run scripts/generate_notebooks.sh to rebuild notebooks."
