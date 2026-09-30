#!/usr/bin/env bash
# Run or resume the basketball pipeline with existing local weights.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

export TORCH_HOME="${TORCH_HOME:-$repo_dir/.cache/torch}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$repo_dir/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_dir/.cache/uv}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is required; install it before running this script.\n' >&2
    exit 1
fi
if [[ ! -x "$repo_dir/.venv/bin/python" ]]; then
    printf 'Prepare the environment first: uv sync --locked --extra loma --extra segmentation\n' >&2
    exit 1
fi

forwarded=("$@")
work_dir="$repo_dir/work/basketball_joint_pipeline_01"
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
        -h|--help) exec uv run --offline --no-sync geosunpre run --help ;;
    esac
    shift
done

# Torch Hub does not honor HF_HUB_OFFLINE; prevent downloads of missing LoMa weights.
for weight in dad.pth dinov2_vitl14_pretrain.pth loma_B.pt; do
    if [[ ! -s "$TORCH_HOME/hub/checkpoints/$weight" ]]; then
        printf 'Missing local weight: %s/hub/checkpoints/%s\n' "$TORCH_HOME" "$weight" >&2
        exit 1
    fi
done

defaults=(--work "$work_dir")
if [[ ! -f "$work_dir/pipeline_config.json" ]]; then
    defaults+=(--input "$repo_dir/input/篮球场"
               --config "$repo_dir/configs/basketball.toml"
               --include-masks --model-cache "$HF_HUB_CACHE")
fi

exec uv run --offline --no-sync geosunpre run "${defaults[@]}" "${forwarded[@]}"
