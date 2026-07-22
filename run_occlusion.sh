#!/usr/bin/env bash
# Occlusion placement: no GenFill, AnyDoor on ref_alpha only, BEHIND restore through holes.
export GENFILL_MODE=occlusion
exec "$(dirname "$0")/run_gradio.sh" "$@"
