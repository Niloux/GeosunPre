#!/usr/bin/env bash
# Run or resume the production pipeline using the prepared local environment.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

if [[ ! -x "$repo_dir/.venv/bin/python" ]]; then
    printf 'Prepare the environment first: uv sync --locked\n' >&2
    exit 1
fi
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$repo_dir/.venv/bin/python" -m geosunpre run "$@"
