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

forwarded=("$@")
work_dir="$repo_dir/work/basketball"
has_input=false
has_config=false
while (($#)); do
    case "$1" in
        --work)
            if (($# < 2)); then
                printf '%s\n' '--work requires a directory' >&2
                exit 2
            fi
            work_dir="$2"
            shift
            ;;
        --work=*) work_dir="${1#*=}" ;;
        --input|--input=*) has_input=true ;;
        --config|--config=*) has_config=true ;;
        -h|--help) exec "$repo_dir/.venv/bin/python" -m geosunpre run --help ;;
    esac
    shift
done

defaults=(--work "$work_dir")
if [[ ! -f "$work_dir/pipeline_config.json" ]]; then
    if [[ "$has_input" == false ]]; then
        defaults+=(--input "$repo_dir/input/篮球场")
        if [[ "$has_config" == false ]]; then
            defaults+=(--config "$repo_dir/configs/basketball.toml")
        fi
    fi
fi
exec "$repo_dir/.venv/bin/python" -m geosunpre run "${defaults[@]}" "${forwarded[@]}"
