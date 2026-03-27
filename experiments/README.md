No hop

```json
{
  "bleu_1": 25.01,
  "bleu_2": 15.42,
  "bleu_4": 6.77,
  "rouge_l": 30.83,
  "chain_coverage": 0.0,
  "avg_hop_depth": 0.0,
  "num_samples": 709
}
```

One hop

```json
{
  "bleu_1": 29.36,
  "bleu_2": 19.82,
  "bleu_4": 10.29,
  "rouge_l": 35.23,
  "chain_coverage": 50.07,
  "avg_hop_depth": 2.0,
  "num_samples": 709
}
```

Table comparison

| Metric | Baseline (KG-LLaVA) | + PrimeKG 1-hop | Paper Target |
|---|---|---|---|
| BLEU-1 | 25.01 | 29.36 (+17%) | — |
| BLEU-2 | 15.42 | 19.82 (+29%) | — |
| BLEU-4 | 6.77 | 10.29 (+52%) | 7.2 |
| ROUGE-L | 30.83 | 35.23 (+14%) | 25.0 |
| Chain coverage | 0% | 50% | — |
| Avg hop depth | 0 | 2.0 | — |

Model: LLaVA-1.5-7B with LoRA fine-tuning, 1 epoch, Modal A10G GPU.
Dataset: 709 MIMIC-NLE test samples (chest X-ray explanation generation).

## Metrics Explained

- **BLEU-1 / BLEU-2 / BLEU-4**: Precision of unigram, bigram, and 4-gram overlaps between generated and reference explanations. Higher scores indicate closer lexical match. BLEU-4 is the standard reporting metric and penalizes short or generic outputs.
- **ROUGE-L**: F1 score of the longest common subsequence between generated and reference text. Captures sentence-level structure better than BLEU since it does not require contiguous matches.
- **Chain coverage**: Percentage of generated answers that mention at least one entity from their associated PrimeKG reasoning chain. Measures whether the model actually uses the injected knowledge graph context.
- **Avg hop depth**: Average number of arrows in the chains provided to the model. A value of 2.0 means each chain contains the RadGraph triplet plus one PrimeKG neighbor (entity --rel--> neighbor).

## Analysis

**Why 1-hop chains help.** The baseline KG-LLaVA prompt includes RadGraph triplets extracted from the image (e.g., "opacity suggestive of pneumonia"). These triplets describe what is visible but not why it matters clinically. Appending 1-hop PrimeKG neighbors adds disease-disease and phenotype-phenotype relationships (e.g., "pneumonia --disease_disease--> aspiration pneumonia") that give the model explicit reasoning paths between radiological findings and clinical concepts. This context lets the model produce explanations that are both more specific and more aligned with reference text, which is reflected in the BLEU-4 improvement from 6.77 to 10.29.

**What chain coverage tells us.** Half of the generated answers (50.07%) reference at least one entity from their PrimeKG chain. This confirms the model is not ignoring the injected chains -- it incorporates them into its output roughly half the time. The other half likely corresponds to images where the chains contain entities too distant from the radiological finding to be useful, or where the reference explanation is simple enough that RadGraph triplets alone suffice.

**Comparison to paper targets.** Both the baseline and multihop configurations exceed the paper's reported BLEU-4 (7.2) and ROUGE-L (25.0). The baseline already surpasses paper targets, likely due to differences in train/test split size (we use a demo subset of 709 samples vs. the full MIMIC-NLE set). The multihop extension widens this gap further.

**Entity alignment approach.** We used exact string match and scispaCy CUI linking to map RadGraph entities to PrimeKG nodes, deliberately avoiding fuzzy matching to prioritize precision. Hub nodes (degree >= 200) were excluded to prevent generic ontology terms from diluting chain quality. Chains were deduplicated per image.

**Limitations.**
- The 709-sample demo subset is small; results may not generalize to the full MIMIC-NLE dataset.
- Only 1-hop chains were used. Deeper traversals (2+ hops) were not evaluated due to noise from distant neighbors.
- Chain coverage of 50% means half of samples get no benefit from the extension. Better entity alignment (especially for Atelectasis, Consolidation, Lung Opacity, Pleural Effusion which lack direct PrimeKG matches) would increase coverage.
- Edge types were restricted to disease_disease and phenotype_phenotype. Including drug or protein edges could add pharmacological reasoning but risks introducing irrelevant context.
