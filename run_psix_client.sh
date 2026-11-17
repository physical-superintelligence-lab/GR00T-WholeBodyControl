#!/usr/bin/env bash
# Usage: ./run_psix_client.sh [--real] --prompt "<task>" [psix_client.py flags, see -h]
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${PSIX_PYTHON:-python}" -u psix_client.py --show-goal "$@"
