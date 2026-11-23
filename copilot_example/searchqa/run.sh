#!/usr/bin/env bash
# Evaluate a Lite skill on SkillOpt's fixed SearchQA split.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${SKILLOPT_LITE_PYTHON:-python3}"
if [[ -z "${SKILLOPT_LITE_PYTHON:-}" && -x "${script_dir}/../../.venv/bin/python" ]]; then
    python_bin="${script_dir}/../../.venv/bin/python"
fi

# Help and argument documentation are available without model credentials.
for arg in "$@"; do
    case "${arg}" in
        -h|--help|--check-data) exec "${python_bin}" -B "${script_dir}/fixed_searchqa.py" "$@" ;;
    esac
done

# Reuse an existing SkillOpt OpenAI-compatible environment on the CPU server.
# Azure and Lite's OpenAI settings still use the original .env bootstrap.
if [[ -z "${SKILLOPT_AUTH_MODE:-}" && -n "${TARGET_OPENAI_COMPATIBLE_BASE_URL:-${OPENAI_COMPATIBLE_BASE_URL:-}}" ]]; then
    export SKILLOPT_AUTH_MODE=openai_compatible
fi
source "${script_dir}/../env.sh" >/dev/null
"${python_bin}" "${script_dir}/fixed_searchqa.py" "$@"
notify_done "SearchQA evaluation finished"
