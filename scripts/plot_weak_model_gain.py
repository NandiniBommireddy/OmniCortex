#!/usr/bin/env python3
"""Visualize whether weaker models benefit disproportionately from augmentation.

This script reads the summary tables in experiments/experiments_2.md, computes
Base -> augmentation gains for Overall score, and plots gain versus base model
strength for RadLex and PrimeKG.

Outputs:
1) a plot showing Base Overall vs Overall gain
2) a JSON summary with correlations and group averages
3) a text file with concise observations derived from the plot
"""

from __future__ import annotations

import argparse
import json
import math
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


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denominator_x == 0 or denominator_y == 0:
        return float("nan")
    return numerator / (denominator_x * denominator_y)


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
    out = {"models": models, "conditions": {}}

    for condition in ["RadLex", "PrimeKG"]:
        rows = []
        for model in models:
            base_overall = judge[model]["Base"]["overall"]
            gain_overall = judge[model][condition]["overall"] - base_overall
            rows.append(
                {
                    "model": model,
                    "base_overall": base_overall,
                    "gain_overall": gain_overall,
                    "base_bleu4": nlg[model]["Base"]["bleu_4"],
                    "gain_bleu4": nlg[model][condition]["bleu_4"] - nlg[model]["Base"]["bleu_4"],
                    "gain_cider": nlg[model][condition]["cider"] - nlg[model]["Base"]["cider"],
                    "gain_rouge_l": nlg[model][condition]["rouge_l"] - nlg[model]["Base"]["rouge_l"],
                }
            )

        base_overalls = [row["base_overall"] for row in rows]
        gains = [row["gain_overall"] for row in rows]
        corr = pearson_r(base_overalls, gains)
        split = sorted(rows, key=lambda row: row["base_overall"])
        midpoint = len(split) // 2
        weak_group = split[:midpoint]
        strong_group = split[midpoint:]

        out["conditions"][condition.lower()] = {
            "rows": rows,
            "pearson_base_vs_gain": corr,
            "mean_gain_weak_half": sum(row["gain_overall"] for row in weak_group) / len(weak_group),
            "mean_gain_strong_half": sum(row["gain_overall"] for row in strong_group) / len(strong_group),
            "median_gain": sorted(gains)[len(gains) // 2],
            "max_gain": max(gains),
            "min_gain": min(gains),
            "weak_half_models": [row["model"] for row in weak_group],
            "strong_half_models": [row["model"] for row in strong_group],
        }

    out["claim"] = {
        "supported_for_primekg": out["conditions"]["primekg"]["pearson_base_vs_gain"] < 0,
        "supported_for_radlex": out["conditions"]["radlex"]["pearson_base_vs_gain"] < 0,
        "primekg_stronger_gap": (
            out["conditions"]["primekg"]["mean_gain_weak_half"]
            > out["conditions"]["primekg"]["mean_gain_strong_half"]
        ),
        "radlex_stronger_gap": (
            out["conditions"]["radlex"]["mean_gain_weak_half"]
            > out["conditions"]["radlex"]["mean_gain_strong_half"]
        ),
    }
    return out


def plot_summary(summary: dict, output_plot: Path):
    conditions = [("radlex", "RadLex", "#d95f02"), ("primekg", "PrimeKG", "#1f77b4")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    model_colors = {model: plt.get_cmap("tab10")(index % 10) for index, model in enumerate(summary["models"])}

    for ax, (key, label, color) in zip(axes, conditions):
        rows = summary["conditions"][key]["rows"]
        xs = [row["base_overall"] for row in rows]
        ys = [row["gain_overall"] for row in rows]
        for row in rows:
            ax.scatter(
                row["base_overall"],
                row["gain_overall"],
                s=70,
                color=model_colors[row["model"]],
                alpha=0.9,
            )

        slope, intercept = linear_fit(xs, ys)
        xs_line = sorted(xs)
        ys_line = [slope * x + intercept for x in xs_line]
        ax.plot(xs_line, ys_line, linestyle="--", color="black", linewidth=1.4)

        x_split = sorted(xs)[len(xs) // 2]
        ax.axvline(x_split, color="gray", linestyle=":", linewidth=1)
        ax.axhline(0, color="gray", linewidth=1)

        ax.set_title(f"{label}: weaker base models gain more")
        ax.set_xlabel("Base Overall score")
        ax.set_ylabel("Overall gain over Base")
        ax.grid(alpha=0.25)

        legend_handles = [
            Line2D([0], [0], color="black", marker="o", linestyle="None", label=f"{label} points", markersize=7),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, label="Linear trend"),
            Line2D([0], [0], color="gray", linestyle=":", linewidth=1, label="Weak/strong split"),
            Line2D([0], [0], color="gray", linestyle="-", linewidth=1, label="Zero gain"),
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
        bbox_to_anchor=(0.83, 0.5),
        frameon=True,
        ncol=1,
    )
    fig.subplots_adjust(right=0.80, wspace=0.18)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)


def build_observations(summary: dict) -> str:
    prime = summary["conditions"]["primekg"]
    radlex = summary["conditions"]["radlex"]
    lines = [
        "Observation from plot and summary",
        "",
        f"PrimeKG Pearson correlation between Base Overall and gain is {prime['pearson_base_vs_gain']:.3f}.",
        f"PrimeKG weak-half mean gain is {prime['mean_gain_weak_half']:.2f}, versus {prime['mean_gain_strong_half']:.2f} for the strong half.",
        f"RadLex Pearson correlation between Base Overall and gain is {radlex['pearson_base_vs_gain']:.3f}.",
        f"RadLex weak-half mean gain is {radlex['mean_gain_weak_half']:.2f}, versus {radlex['mean_gain_strong_half']:.2f} for the strong half.",
        "",
        "Interpretation:",
        "Lower-performing models gain more from augmentation if the correlation is negative and the weak-half average gain is larger than the strong-half average gain.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Visualize whether weak models benefit more from augmentation")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/experiments_2.md"),
        help="Markdown results file",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="plot_weak_model_gain",
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
    output_obs = args.output_dir / f"{args.prefix}_observations.txt"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_summary(summary, output_plot)
    output_obs.write_text(build_observations(summary), encoding="utf-8")

    print(f"Wrote summary JSON: {output_json}")
    print(f"Wrote plot: {output_plot}")
    print(f"Wrote observations: {output_obs}")


if __name__ == "__main__":
    main()
