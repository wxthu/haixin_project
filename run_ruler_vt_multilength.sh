#!/usr/bin/env bash
# 依次对 ruler_niah_vt_multilength 下所有 vt/validation.jsonl 调用评测脚本。
# 使用前请启动 vLLM OpenAI 服务（与 vllm_serve_long_context_openai.py 注释中的端口一致）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/vllm_serve_ruler_vt_openai.py"
DATA_ROOT="${SCRIPT_DIR}/ruler_niah_vt_multilength"

if [[ ! -f "${PY}" ]]; then
  echo "找不到脚本: ${PY}" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "找不到数据目录: ${DATA_ROOT}" >&2
  exit 1
fi

shopt -s nullglob
files=("${DATA_ROOT}"/maxseq_*/vt/validation.jsonl)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "未找到 ${DATA_ROOT}/maxseq_*/vt/validation.jsonl" >&2
  exit 1
fi

IFS=$'\n' files_sorted=($(printf '%s\n' "${files[@]}" | sort -V))
unset IFS

echo "将评测 ${#files_sorted[@]} 个 VT jsonl"
for f in "${files_sorted[@]}"; do
  echo "---- ${f} ----"
  python3 "${PY}" "${f}"
done

echo "全部完成。"
