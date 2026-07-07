#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHONPATH="$ROOT/src:$ROOT/demo_app" python scripts/inject_streamlit_head.py
PYTHONPATH="$ROOT/src:$ROOT/demo_app" exec streamlit run "$ROOT/demo_app/Wikontic.py" "$@"
