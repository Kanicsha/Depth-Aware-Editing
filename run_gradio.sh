#!/usr/bin/env bash
# Launch Gradio object-placement demo.
#
# GenFill / AnyDoor experiment modes (GENFILL_MODE):
#   same_intersection              — SAME ∩ bbox → GenFill; full bbox AnyDoor hint (default)
#   full_bbox                      — entire placement bbox minus FRONT → GenFill
#   same_and_behind_intersection   — C3: (SAME ∪ BEHIND) ∩ bbox → GenFill
#   behind_intersection            — E2: BEHIND ∩ bbox only → GenFill
#   f3                             — F3: SAME → GenFill; AnyDoor hint excludes BEHIND
#   occlusion                      — no GenFill; AnyDoor ref_alpha only; BEHIND holes (see run_occlusion.sh)
#   none                           — skip GenFill (composite still runs)
#
# Examples:
#   GENFILL_MODE=c3 ./run_gradio.sh
#   GENFILL_MODE=e2 ./run_gradio.sh
#   GENFILL_MODE=f3 ./run_gradio.sh
#   ./run_occlusion.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Reuse existing ~/.cache/huggingface by default.
# To store HF + Torch caches inside the repo instead:
#   export DEPTH_EDIT_MODEL_CACHE=1
if [[ "${DEPTH_EDIT_MODEL_CACHE:-}" =~ ^(1|true|yes)$ ]]; then
  export HF_HOME="${HF_HOME:-$ROOT/.model_cache/huggingface}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT/.model_cache/transformers}"
  export TORCH_HOME="${TORCH_HOME:-$ROOT/.model_cache/torch}"
fi

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export GENFILL_MODE="${GENFILL_MODE:-same_intersection}"
exec python gradio_demo_op.py "$@"
