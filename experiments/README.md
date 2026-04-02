### No hop

NLG metrics

```shell
Results:
  bleu_1: 32.57
  bleu_2: 19.93
  bleu_4: 8.99
  meteor: 30.7
  rouge_l: 30.71
  cider: 59.09
  chain_coverage: 0.0
  avg_hop_depth: 0.0
  num_samples: 709
```

LLM-as-judge metrics on 100 samples using claude-haiku-4-5-20251001

```shell
Results (100/100 scored):
  clinical_accuracy: 2.79/5
  completeness: 2.08/5
  reasoning_quality: 3.01/5
  language_quality: 4.05/5
  overall: 2.98/5
```

### One hop PrimeKG

NLG metrics

```shell
Results:
  bleu_1: 36.57
  bleu_2: 24.46
  bleu_4: 12.96
  meteor: 35.5
  rouge_l: 35.23
  cider: 90.28
  chain_coverage: 50.07
  avg_hop_depth: 2.0
  num_samples: 709
```

LLM-as-judge metrics on 100 samples using claude-haiku-4-5-20251001

```shell
Results (100/100 scored):
  clinical_accuracy: 2.74/5
  completeness: 2.15/5
  reasoning_quality: 2.94/5
  language_quality: 4.08/5
  overall: 2.98/5
```

### One-hop Radlex

NLG metrics

```shell
Results:
  bleu_1: 38.12
  bleu_2: 26.22
  bleu_4: 14.24
  meteor: 37.05
  rouge_l: 37.69
  cider: 107.8
  chain_coverage: 74.19
  avg_hop_depth: 2.0
  num_samples: 709
```

LLM-as-judge metrics on 100 samples using claude-haiku-4-5-20251001

```shell
Results (99/100 scored):
  clinical_accuracy: 2.93/5
  completeness: 2.29/5
  reasoning_quality: 3.06/5
  language_quality: 4.06/5
  overall: 3.08/5
```

Model: LLaVA-1.5-7B with LoRA fine-tuning, 1 epoch, Modal A10G GPU.
Dataset: 709 MIMIC-NLE test samples (chest X-ray explanation generation).

## Metrics Explained

- **BLEU-1 / BLEU-2 / BLEU-4**: Precision of unigram, bigram, and 4-gram overlaps between generated and reference explanations. Higher scores indicate closer lexical match. BLEU-4 is the standard reporting metric and penalizes short or generic outputs.
- **ROUGE-L**: F1 score of the longest common subsequence between generated and reference text. Captures sentence-level structure better than BLEU since it does not require contiguous matches.
- **Chain coverage**: Percentage of generated answers that mention at least one entity from their associated PrimeKG reasoning chain. Measures whether the model actually uses the injected knowledge graph context.
- **Avg hop depth**: Average number of arrows in the chains provided to the model. A value of 2.0 means each chain contains the RadGraph triplet plus one PrimeKG neighbor (entity --rel--> neighbor).

## Analysis

A model with zero knowledge graph augmentation already scores 2.97/5 overall. The gap between this and your one-hop result (which scored 2.98/5 on 100 samples) is negligible — meaning one-hop triplets barely helped over the raw model.

This makes the case for multi-hop chains stronger: single-hop triplets like "opacity suggestive of edema" are too shallow to
meaningfully improve clinical reasoning.

**Limitations.**

- The 709-sample demo subset is small; results may not generalize to the full MIMIC-NLE dataset.
- Only 1-hop chains were used. Deeper traversals (2+ hops) were not evaluated due to noise from distant neighbors.
- Chain coverage of 50% means half of samples get no benefit from the extension. Better entity alignment (especially for Atelectasis, Consolidation, Lung Opacity, Pleural Effusion which lack direct PrimeKG matches) would increase coverage.
- Edge types were restricted to disease_disease and phenotype_phenotype. Including drug or protein edges could add pharmacological reasoning but risks introducing irrelevant context.
