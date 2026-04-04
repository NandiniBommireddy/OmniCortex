#!/usr/bin/env python3
"""Compute evaluation metrics for multihop KG-LLaVA outputs.

Metrics: BLEU-1/2/4, METEOR, ROUGE-L, CIDEr, chain coverage, avg hop depth,
         entity recall, hallucination rate.

Usage:
    python scripts/metrics.py \
        --answers tmp/demo/llava_modal_eval/demo_answers.jsonl \
        --references tmp/demo/mimic-nle-test-kg-llava-multihop.json \
        --output tmp/demo/eval_results.json
"""

import argparse
import json
from pathlib import Path

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer


def load_answers(path: Path) -> dict[int, str]:
    """Load model answers from JSONL (question_id -> text)."""
    answers = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                answers[row["question_id"]] = row["text"]
    return answers


def load_references(path: Path) -> list[str]:
    """Load reference NLEs from LLaVA JSON format."""
    data = json.load(open(path))
    refs = []
    for row in data:
        gpt_turn = row["conversations"][1]
        refs.append(gpt_turn["value"])
    return refs


def load_chains(path: Path) -> list[list[str]]:
    """Load chains from LLaVA JSON (extract from human prompt)."""
    data = json.load(open(path))
    all_chains = []
    for row in data:
        prompt = row["conversations"][0]["value"]
        chains = []
        marker = "The multi-hop reasoning chains are: "
        if marker in prompt:
            after = prompt.split(marker, 1)[1]
            # Chains end at ". And for the given image"
            chain_text = after.split(". And for the given image")[0]
            chains = [c.strip() for c in chain_text.split("; ") if c.strip()]
        all_chains.append(chains)
    return all_chains


def compute_bleu(hypotheses: list[str], references: list[str]):
    """Compute BLEU-1, BLEU-2, BLEU-4."""
    smooth = SmoothingFunction().method1
    refs_tok = [[nltk.word_tokenize(ref.lower())] for ref in references]
    hyps_tok = [nltk.word_tokenize(hyp.lower()) for hyp in hypotheses]

    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(refs_tok, hyps_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    return bleu1, bleu2, bleu4


def compute_meteor(hypotheses: list[str], references: list[str]) -> float:
    """Compute average METEOR score."""
    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tok = nltk.word_tokenize(hyp.lower())
        ref_tok = nltk.word_tokenize(ref.lower())
        scores.append(meteor_score([ref_tok], hyp_tok))
    return sum(scores) / len(scores) if scores else 0.0


def compute_cider(hypotheses: list[str], references: list[str]) -> float:
    """Compute CIDEr score using pycocoevalcap."""
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    return score


def compute_rouge_l(hypotheses: list[str], references: list[str]) -> float:
    """Compute average ROUGE-L F1."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for hyp, ref in zip(hypotheses, references):
        score = scorer.score(ref, hyp)
        scores.append(score["rougeL"].fmeasure)
    return sum(scores) / len(scores) if scores else 0.0


def compute_chain_coverage(hypotheses: list[str], chains: list[list[str]]) -> float:
    """% of answers mentioning at least 1 entity from their chains."""
    if not hypotheses:
        return 0.0
    covered = 0
    for hyp, img_chains in zip(hypotheses, chains):
        if not img_chains:
            continue
        hyp_lower = hyp.lower()
        for chain in img_chains:
            # Extract entities from chain: "subj --rel--> obj --edge--> neighbor"
            parts = chain.split(" --")
            for part in parts:
                entity = part.split("--> ")[-1].strip() if "--> " in part else part.strip()
                if entity.lower() in hyp_lower:
                    covered += 1
                    break
            else:
                continue
            break
    return covered / len(hypotheses)


# Clinical entity vocabulary: 10 MIMIC-NLE diagnoses + common radiological findings
CLINICAL_ENTITIES = [
    # MIMIC-NLE diagnoses
    "atelectasis", "consolidation", "edema", "enlarged cardiomediastinum",
    "lung lesion", "lung opacity", "pleural effusion", "pleural other",
    "pneumonia", "pneumothorax",
    # Common radiological findings
    "opacity", "effusion", "infiltrate", "haziness", "fluid", "congestion",
    "cardiomegaly", "blunting", "thickening", "nodule", "mass", "interstitial",
    "airspace", "vascular", "markings", "lucency", "density", "infiltration",
]


def extract_entities(text: str) -> set[str]:
    """Extract clinical entities present in text via substring matching."""
    text_lower = text.lower()
    return {e for e in CLINICAL_ENTITIES if e in text_lower}


def compute_entity_recall(hypotheses: list[str], references: list[str]) -> float:
    """Avg fraction of reference entities that appear in the hypothesis."""
    scores = []
    for hyp, ref in zip(hypotheses, references):
        ref_entities = extract_entities(ref)
        if not ref_entities:
            continue
        found = ref_entities & extract_entities(hyp)
        scores.append(len(found) / len(ref_entities))
    return sum(scores) / len(scores) if scores else 0.0


def compute_hallucination_rate(hypotheses: list[str], references: list[str]) -> float:
    """Avg fraction of hypothesis entities NOT present in the reference."""
    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_entities = extract_entities(hyp)
        if not hyp_entities:
            continue
        ref_entities = extract_entities(ref)
        hallucinated = hyp_entities - ref_entities
        scores.append(len(hallucinated) / len(hyp_entities))
    return sum(scores) / len(scores) if scores else 0.0


def compute_avg_hop_depth(chains: list[list[str]]) -> float:
    """Average number of hops across all chains (arrows per chain)."""
    total_hops = 0
    total_chains = 0
    for img_chains in chains:
        for chain in img_chains:
            hops = chain.count("-->")
            total_hops += hops
            total_chains += 1
    return total_hops / total_chains if total_chains else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate multihop KG-LLaVA outputs")
    parser.add_argument("--answers", required=True, type=Path, help="Model answers JSONL")
    parser.add_argument("--references", required=True, type=Path, help="Reference LLaVA JSON")
    parser.add_argument("--output", type=Path, default=Path("tmp/demo/eval_results.json"))
    args = parser.parse_args()

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    print("Loading data...")
    answers = load_answers(args.answers)
    references = load_references(args.references)
    chains = load_chains(args.references)

    # Align answers with references by index
    hypotheses = [answers.get(i, "") for i in range(len(references))]
    num_samples = len(hypotheses)
    print(f"  {num_samples} samples, {sum(1 for h in hypotheses if h)} with answers")

    print("Computing metrics...")
    bleu1, bleu2, bleu4 = compute_bleu(hypotheses, references)
    meteor = compute_meteor(hypotheses, references)
    rouge_l = compute_rouge_l(hypotheses, references)
    cider = compute_cider(hypotheses, references)
    chain_cov = compute_chain_coverage(hypotheses, chains)
    avg_depth = compute_avg_hop_depth(chains)
    entity_recall = compute_entity_recall(hypotheses, references)
    hallucination_rate = compute_hallucination_rate(hypotheses, references)

    results = {
        "bleu_1": round(bleu1 * 100, 2),
        "bleu_2": round(bleu2 * 100, 2),
        "bleu_4": round(bleu4 * 100, 2),
        "meteor": round(meteor * 100, 2),
        "rouge_l": round(rouge_l * 100, 2),
        "cider": round(cider * 100, 2),
        "chain_coverage": round(chain_cov * 100, 2),
        "avg_hop_depth": round(avg_depth, 2),
        "entity_recall": round(entity_recall * 100, 2),
        "hallucination_rate": round(hallucination_rate * 100, 2),
        "num_samples": num_samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
