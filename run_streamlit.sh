#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python scripts/inject_streamlit_head.py
PYTHONPATH=src exec streamlit run Wikontic.py "$@"
