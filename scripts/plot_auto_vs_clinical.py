#!/usr/bin/env python3
"""Visualize whether automatic metrics improve more than clinical quality.

This script reads the summary tables in experiments/experiments_2.md, computes
Base -> augmentation gains for automatic metrics and Overall score, and plots
CIDEr gain versus Overall gain for RadLex and PrimeKG.

Outputs:
1) a plot showing automatic-metric gain vs clinical-gain
2) a JSON summary with medians and counts
3) a text file with concise observations derived from the plot
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def find_line_index(lines: list[str], needle: str) -> int:
    for index, line in enumerate(lines):
        if line.strip() == needle:
            return index
    return -1


def parse_float(cell: str) -> float:
    cleaned = re.sub(r"\*", "", cell).strip()
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0].strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ValueError(f"Cannot parse numeric value from: {cell}")
    return float(match.group(0))


def normalize_model_name(cell: str) -> str:
    return re.sub(r"\*", "", cell).strip()


def parse_table(lines: list[str], section_title: str, metric_map: dict[str, str]):
    section_idx = find_line_index(lines, section_title)
    if section_idx < 0:
        raise ValueError(f"Missing section: {section_title}")

    header_idx = -1
    for index in range(section_idx + 1, min(len(lines), section_idx + 30)):
        if lines[index].strip().startswith("| Model"):
            header_idx = index
            break
    if header_idx < 0:
        raise ValueError(f"Missing table header under: {section_title}")

    rows = {}
    last_model = None

    for index in range(header_idx + 2, len(lines)):
        raw = lines[index].strip()
        if not raw.startswith("|"):
            if raw.startswith("## "):
                break
            continue

        parts = [part.strip() for part in raw.split("|")[1:-1]]
        if len(parts) < 2:
            continue

        model_cell = parts[0]
        model_name = normalize_model_name(model_cell) if model_cell else last_model
        if not model_name:
            continue

        condition = parts[1]
        if condition not in {"Base", "RadLex", "PrimeKG"}:
            continue

        last_model = model_name
        metric_cells = parts[2:]
        metrics = {}
        for (_, key), value in zip(metric_map.items(), metric_cells):
            metrics[key] = parse_float(value)
        rows.setdefault(model_name, {})[condition] = metrics

    return rows


def parse_markdown_tables(md_text: str):
    lines = md_text.splitlines()
    nlg_headers = {
        "BLEU-4": "bleu_4",
        "METEOR": "meteor",
        "ROUGE-L": "rouge_l",
        "CIDEr": "cider",
        "Entity Recall": "entity_recall",
        "Hallucination Rate": "hallucination_rate",
    }
    judge_headers = {
        "Clinical Accuracy": "clinical_accuracy",
        "Completeness": "completeness",
        "Reasoning Quality": "reasoning_quality",
        "Language Quality": "language_quality",
        "Overall": "overall",
    }
    nlg = parse_table(lines, "## NLG Metrics (All Models)", nlg_headers)
    judge = parse_table(lines, "## LLM-as-Judge Metrics (All Models)", judge_headers)
    return nlg, judge


def linear_fit(xs: list[float], ys: list[float]):
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0, mean_y
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


def build_summary(nlg: dict, judge: dict):
    models = sorted(judge.keys())
    summary = {"models": models, "conditions": {}}

    for condition in ["RadLex", "PrimeKG"]:
        rows = []
        for model in models:
            rows.append(
                {
                    "model": model,
                    "base_overall": judge[model]["Base"]["overall"],
                    "gain_overall": judge[model][condition]["overall"] - judge[model]["Base"]["overall"],
                    "gain_cider": nlg[model][condition]["cider"] - nlg[model]["Base"]["cider"],
                    "gain_bleu4": nlg[model][condition]["bleu_4"] - nlg[model]["Base"]["bleu_4"],
                    "gain_meteor": nlg[model][condition]["meteor"] - nlg[model]["Base"]["meteor"],
                    "gain_rouge_l": nlg[model][condition]["rouge_l"] - nlg[model]["Base"]["rouge_l"],
                    "gain_entity_recall": nlg[model][condition]["entity_recall"] - nlg[model]["Base"]["entity_recall"],
                    "gain_hallucination": nlg[model]["Base"]["hallucination_rate"]
                    - nlg[model][condition]["hallucination_rate"],
                }
            )

        cider_gains = [row["gain_cider"] for row in rows]
        overall_gains = [row["gain_overall"] for row in rows]
        median_cider = sorted(cider_gains)[len(cider_gains) // 2]
        median_overall = sorted(overall_gains)[len(overall_gains) // 2]
        small_overall = sum(1 for gain in overall_gains if 0.05 <= gain <= 0.15)
        nonpositive_overall = sum(1 for gain in overall_gains if gain <= 0)

        summary["conditions"][condition.lower()] = {
            "rows": rows,
            "median_cider_gain": median_cider,
            "median_overall_gain": median_overall,
            "mean_cider_gain": sum(cider_gains) / len(cider_gains),
            "mean_overall_gain": sum(overall_gains) / len(overall_gains),
            "small_overall_gain_count": small_overall,
            "nonpositive_overall_gain_count": nonpositive_overall,
            "supported_typically": median_cider >= 10 and median_overall <= 0.15,
        }

    summary["claim"] = {
        "primekg_supported": summary["conditions"]["primekg"]["supported_typically"],
        "radlex_supported": summary["conditions"]["radlex"]["supported_typically"],
        "overall_supported": summary["conditions"]["primekg"]["supported_typically"]
        and summary["conditions"]["radlex"]["supported_typically"],
    }
    return summary


def plot_summary(summary: dict, output_plot: Path):
    conditions = [("radlex", "RadLex", "#d95f02"), ("primekg", "PrimeKG", "#1f77b4")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    model_colors = {model: plt.get_cmap("tab10")(index % 10) for index, model in enumerate(summary["models"])}

    for ax, (key, label, _) in zip(axes, conditions):
        rows = summary["conditions"][key]["rows"]
        xs = [row["gain_cider"] for row in rows]
        ys = [row["gain_overall"] for row in rows]

        for row in rows:
            ax.scatter(
                row["gain_cider"],
                row["gain_overall"],
                s=70,
                color=model_colors[row["model"]],
                alpha=0.9,
            )

        slope, intercept = linear_fit(xs, ys)
        xs_line = sorted(xs)
        ys_line = [slope * x + intercept for x in xs_line]
        ax.plot(xs_line, ys_line, linestyle="--", color="black", linewidth=1.4)

        ax.axhline(0, color="gray", linewidth=1)
        ax.axhspan(0.05, 0.15, color="#e8f5e9", alpha=0.5)

        ax.set_title(f"{label}: automatic gains vs clinical gains")
        ax.set_xlabel("CIDEr gain over Base")
        ax.set_ylabel("Overall gain over Base")
        ax.grid(alpha=0.25)

        legend_handles = [
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, label="Trend line"),
            Line2D([0], [0], color="gray", linestyle="-", linewidth=1, label="Zero gain"),
            Line2D([0], [0], color="#e8f5e9", marker="s", linestyle="None", label="Marginal clinical gain band"),
        ]
        ax.legend(handles=legend_handles, loc="best", frameon=True)

    model_handles = [
        Line2D([0], [0], marker="o", color="w", label=model, markerfacecolor=model_colors[model], markersize=8)
        for model in summary["models"]
    ]
    fig.legend(
        handles=model_handles,
        title="Model color",
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        frameon=True,
        ncol=1,
    )
    fig.subplots_adjust(right=0.81, wspace=0.20)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)


def plot_all_nlg_vs_overall(summary: dict, output_plot: Path):
    metric_specs = [
        ("gain_bleu4", "BLEU-4 gain", False),
        ("gain_meteor", "METEOR gain", False),
        ("gain_rouge_l", "ROUGE-L gain", False),
        ("gain_cider", "CIDEr gain", False),
        ("gain_entity_recall", "Entity Recall gain", False),
        ("gain_hallucination", "Hallucination improvement", True),
    ]

    rows = []
    for key, label, marker in [("radlex", "RadLex", "o"), ("primekg", "PrimeKG", "^")]:
        for row in summary["conditions"][key]["rows"]:
            point = dict(row)
            point["condition"] = label
            point["marker"] = marker
            rows.append(point)

    model_colors = {model: plt.get_cmap("tab10")(index % 10) for index, model in enumerate(summary["models"])}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)

    for ax, (metric_key, title, is_improvement) in zip(axes.flat, metric_specs):
        xs = [row[metric_key] for row in rows]
        ys = [row["gain_overall"] for row in rows]

        for row in rows:
            ax.scatter(
                row[metric_key],
                row["gain_overall"],
                s=65,
                color=model_colors[row["model"]],
                marker=row["marker"],
                alpha=0.9,
            )

        slope, intercept = linear_fit(xs, ys)
        xs_line = sorted(xs)
        ys_line = [slope * x + intercept for x in xs_line]
        ax.plot(xs_line, ys_line, linestyle="--", color="black", linewidth=1.2)

        ax.axhline(0, color="gray", linewidth=1)
        ax.axhspan(0.05, 0.15, color="#e8f5e9", alpha=0.45)
        ax.grid(alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("NLG metric gain over Base" if not is_improvement else "Lower hallucination over Base")
        ax.set_ylabel("Overall gain over Base")

    semantic_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=7, label="RadLex"),
        Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=7, label="PrimeKG"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Trend line"),
        Line2D([0], [0], color="gray", linestyle="-", linewidth=1, label="Zero gain"),
        Line2D([0], [0], color="#e8f5e9", marker="s", linestyle="None", label="Marginal clinical gain band"),
    ]

    model_handles = [
        Line2D([0], [0], marker="o", color="w", label=model, markerfacecolor=model_colors[model], markersize=7)
        for model in summary["models"]
    ]

    fig.legend(
        handles=semantic_handles,
        title="Markers and lines",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
    )
    fig.legend(
        handles=model_handles,
        title="Model color",
        loc="center right",
        bbox_to_anchor=(0.98, 0.45),
        frameon=True,
        ncol=1,
    )
    fig.subplots_adjust(right=0.80, wspace=0.20, hspace=0.28)
    fig.suptitle("All NLG metric gains vs clinical Overall gain", fontsize=14)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)


def build_observations(summary: dict) -> str:
    prime = summary["conditions"]["primekg"]
    radlex = summary["conditions"]["radlex"]
    lines = [
        "Observation from plot and summary",
        "",
        f"PrimeKG median CIDEr gain is {prime['median_cider_gain']:.2f}, while median Overall gain is {prime['median_overall_gain']:.2f}.",
        f"RadLex median CIDEr gain is {radlex['median_cider_gain']:.2f}, while median Overall gain is {radlex['median_overall_gain']:.2f}.",
        f"PrimeKG small Overall gains (0.05 to 0.15) occur in {prime['small_overall_gain_count']}/8 models.",
        f"RadLex small Overall gains (0.05 to 0.15) occur in {radlex['small_overall_gain_count']}/8 models.",
        "",
        "Interpretation:",
        "The plot supports the claim if automatic metric gains are much larger than clinical gains and most points sit near the bottom-right: large x-values with small y-values.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Visualize whether automatic metrics improve more than clinical quality")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/experiments_2.md"),
        help="Markdown results file",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="plot_auto_vs_clinical",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Output directory",
    )
    args = parser.parse_args()

    md_text = args.input.read_text(encoding="utf-8")
    nlg, judge = parse_markdown_tables(md_text)
    summary = build_summary(nlg, judge)

    output_json = args.output_dir / f"{args.prefix}_summary.json"
    output_plot = args.output_dir / f"{args.prefix}_plot.png"
    output_all_nlg_plot = args.output_dir / f"{args.prefix}_all_nlg_plot.png"
    output_obs = args.output_dir / f"{args.prefix}_observations.txt"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_summary(summary, output_plot)
    plot_all_nlg_vs_overall(summary, output_all_nlg_plot)
    output_obs.write_text(build_observations(summary), encoding="utf-8")

    print(f"Wrote summary JSON: {output_json}")
    print(f"Wrote plot: {output_plot}")
    print(f"Wrote all-NLG plot: {output_all_nlg_plot}")
    print(f"Wrote observations: {output_obs}")


if __name__ == "__main__":
    main()
