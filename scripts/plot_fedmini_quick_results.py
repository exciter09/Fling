import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value in (None, ""):
        return None
    return float(value)


def build_series(rows):
    rounds = [int(row["round"]) for row in rows]
    cumulative_mb = [to_float(row["cumulative_trans_cost_mb"]) for row in rows]
    tested_rounds = []
    tested_acc = []
    for row in rows:
        acc = to_float(row.get("after_aggregation_test.test_acc"))
        if acc is not None:
            tested_rounds.append(int(row["round"]))
            tested_acc.append(acc * 100.0)
    return dict(rounds=rounds, cumulative_mb=cumulative_mb, tested_rounds=tested_rounds, tested_acc=tested_acc)


def pick_font():
    candidates = [
        "Songti SC",
        "STSong",
        "PingFang SC",
        "Heiti SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def main():
    parser = argparse.ArgumentParser(description="Plot paper-style quick FedMini figures.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dirichlet_rows = load_rows(input_root / "cifar100/dirichlet/value_0p1/seed_0/round_metrics.csv")
    pathological_rows = load_rows(input_root / "cifar100/pathological/value_10/seed_0/round_metrics.csv")

    dirichlet = build_series(dirichlet_rows)
    pathological = build_series(pathological_rows)

    font_name = pick_font()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=180)
    fig.patch.set_facecolor("white")

    paper_blue = "#2f5aa8"
    paper_red = "#b4483a"
    paper_gold = "#c28f2c"
    grid_color = "#d9d9d9"

    ax = axes[0]
    ax.plot(
        dirichlet["tested_rounds"],
        dirichlet["tested_acc"],
        color=paper_blue,
        linewidth=2.4,
        marker="o",
        markersize=4.5,
        label="FedMini Dirichlet α=0.1",
    )
    ax.plot(
        pathological["tested_rounds"],
        pathological["tested_acc"],
        color=paper_red,
        linewidth=2.4,
        marker="s",
        markersize=4.5,
        label="FedMini Pathological",
    )
    ax.axvline(94, color=paper_gold, linestyle="--", linewidth=1.6, label="Freeze starts (round 94)")
    ax.set_title("准确率-通信轮次仿真图", fontsize=14, pad=10)
    ax.set_xlabel("通信轮次", fontsize=11)
    ax.set_ylabel("平均测试准确率 (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(5, 60)
    ax.grid(True, linestyle="--", linewidth=0.8, color=grid_color, alpha=0.9)
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    ax = axes[1]
    ax.plot(
        dirichlet["rounds"],
        dirichlet["cumulative_mb"],
        color=paper_blue,
        linewidth=2.4,
        label="FedMini cumulative cost",
    )
    ax.axvline(94, color=paper_gold, linestyle="--", linewidth=1.6, label="Freeze starts (round 94)")
    ax.set_title("通信开销-通信轮次仿真图", fontsize=14, pad=10)
    ax.set_xlabel("通信轮次", fontsize=11)
    ax.set_ylabel("累计通信开销 (MB)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle="--", linewidth=0.8, color=grid_color, alpha=0.9)
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#444444")
        axis.spines["bottom"].set_color("#444444")

    fig.suptitle("FedMini 快速实验结果（CIFAR-100，论文风格复现图）", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
