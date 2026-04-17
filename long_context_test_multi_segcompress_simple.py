from datasets import load_dataset
import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from llmlingua import PromptCompressor

from compress_conversation_value import parse_conversation_value

'''
python3 -m vllm.entrypoints.openai.api_server  \
--host 0.0.0.0  --port 8090 --max-num-seqs 4  --max-model-len 30000 \
--model Qwen/Qwen3-32B-AWQ
'''

URL = "http://0.0.0.0:8090/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "YzQzNDA1Y2VjOWZmYTk0MmJlYTNlMTljMWM0MzdjMjM3ZDhjZTgzMQ==",
}
example = '["我的左脚","血色将至","最后的莫希干人","纽约黑帮","林肯","血色黑金","最后一个莫西干人"]'
llm_lingua = PromptCompressor("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True)

# 并发配置
MAX_WORKERS = 4  # 根据服务器承受能力调整，建议 <= max-num-seqs
TIMEOUT = 120  # 请求超时时间（秒）


def get_llm_response(query):
    """
    仅对 query 做预处理：先用 parse_conversation_value 分段，
    固定段保持不变，仅压缩 [SEP] 后每个编号文档段，然后再拼回完整 prompt。
    其余逻辑不改。
    """
    segments = parse_conversation_value(query)

    compressed_parts = []
    rr_sum = 0.0
    rr_w_sum = 0.0

    for seg in segments:
        if not seg.compressible:
            compressed_parts.append(seg.text)
            continue

        result = llm_lingua.compress_prompt(
            seg.text,
            question="question",
            rate=0.5,
            adaptive_compression=True,
            adaptive_strategy="elbow",
            # adaptive_strategy="mass",
            # adaptive_mass_alpha=0.9,
            condition_in_question="after_condition",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.0,
            condition_compare=True,
            context_budget="+100",
            rank_method="longllmlingua",
        )
        compressed_parts.append(result["compressed_prompt"])

        rr = result.get("compression_rate", 0.0)
        w = len(seg.text)
        rr_sum += rr * w
        rr_w_sum += w

    query = "".join(compressed_parts)
    rr = (rr_sum / rr_w_sum) if rr_w_sum > 0 else 0.0

    payload_json = {
        "stream": False,
        "messages": [
            {
                "content": f"{query} \n ### 相关媒资提取，输出格式：[名称1,名称2], 输出示例：{example}",
                "role": "user",
            },
        ],
        "max_tokens": 100,
        "ignore_eos": False,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(URL, headers=HEADERS, json=payload_json, timeout=TIMEOUT)
    return response, rr


def process_single_item(data, index):
    """处理单条数据，返回 (index, iou) 保证结果顺序"""
    response_content = None
    try:
        answer = json.loads(data["chosen"]["value"])
        answer_list = [item.split("#")[1].split("|")[0] for item in answer]

        response, rr = get_llm_response(query=data["conversations"][0]["value"])
        response.raise_for_status()
        response_content = response.json()["choices"][0]["message"]["content"]
        response_list = json.loads(response_content)

        intersection = set(answer_list) & set(response_list)
        union = set(answer_list) | set(response_list)

        iou = len(intersection) / len(union) if len(union) > 0 else 1.0
        return index, iou, rr
    except Exception as e:
        snippet = (response_content[:300] + "…") if response_content else None
        print(f"Error at index {index}: {e!r}" + (f" | model_output={snippet!r}" if snippet else ""))
        # 返回 (index, iou, rr) 保持与成功路径一致，rr 置为 0.0 表示缺失
        return index, 0.0, 0.0


def normalize_worker_result(result):
    """
    兼容历史返回格式，统一成 (idx, iou, rr)。
    - 旧格式: (idx, iou)
    - 新格式: (idx, iou, rr)
    """
    if not isinstance(result, tuple):
        raise ValueError(f"Worker returned non-tuple result: {type(result).__name__}")
    if len(result) == 3:
        return result
    if len(result) == 2:
        idx, iou = result
        return idx, iou, 0.0
    raise ValueError(f"Worker returned unexpected tuple length: {len(result)}")


def main():
    data_path = "train_data_summary_doubao_ORPO_0722_complex_cut18000.json"
    dataset = load_dataset("json", data_files=data_path)["train"]  # .select(range(4))

    # 预分配结果列表，保证顺序
    iou_results = [None] * len(dataset)
    rr_results = [None] * len(dataset)

    # 使用线程池并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_single_item, data, idx): idx for idx, data in enumerate(dataset)
        }

        for future in tqdm(as_completed(future_to_idx), total=len(dataset), desc="Processing"):
            submit_idx = future_to_idx[future]
            try:
                idx, iou, rr = normalize_worker_result(future.result())
                iou_results[idx] = iou
                rr_results[idx] = rr
            except Exception as e:
                print(f"Future failed at index {submit_idx}: {e!r}")
                iou_results[submit_idx] = 0.0
                rr_results[submit_idx] = 0.0

    # 计算平均 IoU（过滤 None 值）
    valid_results = [r for r in iou_results if r is not None]
    rr_results_valid = [r for r in rr_results if r is not None]
    avg_iou = sum(valid_results) / len(valid_results) if valid_results else 0.0
    avg_rr = sum(rr_results_valid) / len(rr_results_valid) if rr_results_valid else 0.0
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Compression Rate: {avg_rr:.4f}")


if __name__ == "__main__":
    main()

