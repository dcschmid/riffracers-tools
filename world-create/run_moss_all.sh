#!/usr/bin/env bash
set -euo pipefail
tool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
python_bin="${RIFF_PYTHON:-$HOME/.local/share/local-image-ai/venv/bin/python}"
exec "$python_bin" "$tool_dir/run_batch.py" moss "$@"
