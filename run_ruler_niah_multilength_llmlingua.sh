#!/usr/bin/env bash
# 依次对 ruler_niah_vt_multilength 下所有 niah_single_1/validation.jsonl 调用
# vllm_serve_ruler_niah_openai_llmlingua.py（首末行保留、中间 LLMLingua 压缩）。
# 使用前请启动 vLLM OpenAI 兼容服务: vllm.entrypoints.openai.api_server
# 依赖: Python 环境已安装 torch、llmlingua、openai、tqdm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/vllm_serve_ruler_niah_openai_llmlingua.py"
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
files=("${DATA_ROOT}"/maxseq_*/niah_single_1/validation.jsonl)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "未找到 ${DATA_ROOT}/maxseq_*/niah_single_1/validation.jsonl" >&2
  exit 1
fi

IFS=$'\n' files_sorted=($(printf '%s\n' "${files[@]}" | sort -V))
unset IFS

echo "将评测 ${#files_sorted[@]} 个 NIAH jsonl（LLMLingua 中间段压缩）"
for f in "${files_sorted[@]}"; do
  echo "---- ${f} ----"
  python3 "${PY}" "${f}" "$@"
done

echo "全部完成。"
