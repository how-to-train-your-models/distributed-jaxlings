#!/usr/bin/env bash
# Convert src/*.py → chapters/*.ipynb using jupytext.
# Outputs are placed in chapters/ to preserve Colab links and existing structure.
# Usage:
#   ./scripts/generate_notebooks.sh          # regenerate all
#   ./scripts/generate_notebooks.sh 01 03    # regenerate specific chapters by number

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
SRC="$ROOT/src"
CHAPTERS="$ROOT/chapters"

JUPYTEXT="uv run jupytext"

mkdir -p "$CHAPTERS"

if [[ $# -eq 0 ]]; then
    # Regenerate all chapters
    files=("$SRC"/chapter_*.py)
else
    # Regenerate only specified chapter numbers
    files=()
    for num in "$@"; do
        matches=("$SRC"/chapter_"${num}"_*.py)
        if [[ ${#matches[@]} -eq 0 || ! -e "${matches[0]}" ]]; then
            echo "Warning: no source file found for chapter ${num}, skipping."
        else
            files+=("${matches[@]}")
        fi
    done
fi

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No .py source files found in $SRC. Run scripts/init_py_sources.sh first."
    exit 1
fi

for py in "${files[@]}"; do
    name="$(basename "$py" .py)"
    out="$CHAPTERS/${name}.ipynb"
    echo "Generating chapters/${name}.ipynb"
    $JUPYTEXT --to notebook --output "$out" "$py"
done

echo ""
echo "Done. Generated ${#files[@]} notebook(s) in chapters/."
