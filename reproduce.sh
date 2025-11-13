#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python reproduce.py
