import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_data(csv_path):
    # 结构：{batch_size: {input_length: (attn, ffn, other)}}
    data = defaultdict(dict)
    input_lengths_set = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bs = int(row["batch_size"])
            il = int(row["input_length"])
            attn = float(row["attention_ratio"])
            ffn = float(row["ffn_ratio"])
            other = float(row["other_ratio"])
            data[bs][il] = (attn, ffn, other)
            input_lengths_set.add(il)

    input_lengths = sorted(input_lengths_set)
    batch_sizes = sorted(data.keys())
    return data, batch_sizes, input_lengths


def plot_attention_lines(data, batch_sizes, input_lengths, out_path="attention_ratio_lines.png"):
    plt.figure(figsize=(8, 5))

    for bs in batch_sizes:
        attn_vals = [data[bs][il][0] for il in input_lengths]
        plt.plot(input_lengths, attn_vals, marker="o", label=f"batch_size={bs}")

    plt.xlabel("input_length")
    plt.ylabel("attention_ratio (%)")
    plt.title("Attention Ratio vs Input Length (per batch_size)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"保存折线图到: {out_path}")


def plot_stacked_bars(
    data,
    batch_sizes,
    input_lengths,
    out_path="ratio_stacked_bars_2x2.png",
):
    # 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    width = 0.6  # 每个 input_length 的条宽

    for idx, bs in enumerate(batch_sizes):
        ax = axes[idx]
        attn_vals = [data[bs][il][0] for il in input_lengths]
        ffn_vals = [data[bs][il][1] for il in input_lengths]
        other_vals = [data[bs][il][2] for il in input_lengths]

        x = np.arange(len(input_lengths))

        # 堆叠条形图
        ax.bar(x, attn_vals, width, label="Attention")
        ax.bar(x, ffn_vals, width, bottom=attn_vals, label="Linear")
        bottom_other = np.array(attn_vals) + np.array(ffn_vals)
        ax.bar(x, other_vals, width, bottom=bottom_other, label="Other")

        ax.set_title(f"batch_size={bs}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(il) for il in input_lengths], rotation=45)
        ax.set_xlabel("input_length")
        if idx % 2 == 0:
            ax.set_ylabel("ratio (%)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # 在条形图上加粗标注 attention_ratio（各占比是百分数）
        for xi, a, f, o in zip(x, attn_vals, ffn_vals, other_vals):
            ax.text(
                xi,
                a / 2.0,
                f"{a:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
            )

    # 统一图例放外面
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"保存堆叠条形图到: {out_path}")


def main():
    csv_path = "vllm_decode_sweep.csv"
    data, batch_sizes, input_lengths = load_data(csv_path)

    # 1) 折线图：不同 batch_size 的 attention_ratio 曲线
    plot_attention_lines(data, batch_sizes, input_lengths)

    # 2) 2x2 堆叠条形图：每个 batch_size 一个子图
    if len(batch_sizes) != 4:
        print("警告：当前 batch_sizes 数量不是 4 个，2x2 子图布局可能不完全匹配。")
    plot_stacked_bars(data, batch_sizes, input_lengths)


if __name__ == "__main__":
    main()

