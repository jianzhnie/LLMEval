#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: scripts/run.sh [entrypoint] [args...]

Run one repository-owned orchestration entrypoint. With no argument, the
standard data-parallel evaluation launcher is used.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

entrypoint="${1:-${SCRIPT_DIR}/data_parallel_infer/start_parallel_eval.sh}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ "${entrypoint}" != /* ]]; then
    entrypoint="${PROJECT_ROOT}/${entrypoint}"
fi
if [[ ! -f "${entrypoint}" ]]; then
    printf 'Entrypoint not found: %s\n' "${entrypoint}" >&2
    exit 1
fi

exec bash "${entrypoint}" "$@"
