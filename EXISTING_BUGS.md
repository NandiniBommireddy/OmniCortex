# OmniCortex — KG-LLaVA Reproducibility

## Project Overview

Reproduces **KG-LLaVA** from "LLaVA Needs More Knowledge: Retrieval Augmented Natural Language Generation with Knowledge Graph for Explaining Thoracic Pathologies" (AAAI 2025, Kyung Hee University).

**Architecture:**
- Vision encoder: MedCLIP ViT (512-dim embeddings)
- KG source: RadGraph `suggestive_of` triplets from MIMIC-CXR reports
- Retrieval: FAISS IndexFlatIP, k=7 neighbors (cross-modal: image query → text-embedded captions)
- LLM: LLaVA-1.5-7B with LoRA fine-tuning (lora_r=64, lora_alpha=128)
- Dataset: MIMIC-NLE (38,003 NLEs, 10 diagnoses, 3 certainty levels)
- **Target metrics:** AUC 83.0, BLEU-4 7.2, CIDEr 62.2, ROUGE-L 25.0

---

## Known Bugs (Fix Before Running)

1. **Eval runs on training data** (`scripts/modal_demo_eval_llava.py` line 13): `LOCAL_DATA` points to `mimic-nle-train-kg-llava.json`. Should point to the dev/test split JSON.

2. **Triplets extracted from NLE sentences, not full reports** (`scripts/extract_radgraph_triplets.py`): Paper builds KG from full MIMIC-CXR reports; current code processes single NLE sentences → sparse triplets. ~50% of eval entries have empty triplet context.

3. **No metric computation**: Eval script only generates `demo_answers.jsonl`; does not compute BLEU, ROUGE, CIDEr, or AUC.