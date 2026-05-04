#!/usr/bin/env bash
# Convert src/chapters/*.py → notebooks/*.ipynb using jupytext.
# Outputs are placed in notebooks/ to preserve Colab links and existing structure.
# Usage:
#   ./scripts/generate_notebooks.sh          # regenerate all
#   ./scripts/generate_notebooks.sh 01 03    # regenerate specific chapters by number

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
SRC="$ROOT/src/chapters"
CHAPTERS="$ROOT/notebooks"

JUPYTEXT="uv run jupytext"

mkdir -p "$CHAPTERS"

# Always sync the shared judge module so notebooks can import it
cp "$ROOT/src/judge.py" "$CHAPTERS/judge.py"

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
    echo "Generating notebooks/${name}.ipynb"
    $JUPYTEXT --to notebook --output "$out" "$py"
done

echo ""
echo "Done. Generated ${#files[@]} notebook(s) in notebooks/."
