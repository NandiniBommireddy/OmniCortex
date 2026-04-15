## Problem Statement

This work reproduces and extends KG-LLaVA (AAAI 2025) — a system for generating Natural Language Explanations (NLEs) for chest X-ray diagnoses.

The core problem: LLaVA can identify pathologies but lacks clinical reasoning chains. The baseline approach retrieves RadGraph triplets ("opacity suggestive of pneumonia") but does not explain why that relationship matters clinically.

This extension adds knowledge graph traversal (PrimeKG and RadLex) to inject structured reasoning chains into prompts, giving the model deeper clinical context.

---

## Dataset

### Source: MIMIC-CXR-JPG v2.1.0 (PhysioNet) + MIMIC-NLE

| Split | Samples |
|-------|---------|
| Train | 36,731 |
| Test  | 709 (demo eval) |

Target variables: 10 diagnoses × 3-state certainty (negative / uncertain / positive)

- `scripts/build_demo_llava_json.py:12-23` — DIAGNOSIS_LIST
- `scripts/build_demo_llava_json.py:25` — CERTAINTY_LIST
- `scripts/build_demo_llava_json.py:27` — QUESTION_TEMPLATE

### Preprocessing Pipeline

1. **RadGraph triplet extraction** — `scripts/extract_radgraph_triplets.py`
   - Gives structured facts from free-text reports
   - Model: modern-radgraph-xl, batch=4, workers=4
   - Output: `"subject suggestive_of object"` strings per report

2. **FAISS datastore** — `scripts/build_demo_datastore.py`
   - Fast retrieval at inference
   - MedCLIP 512-dim embeddings, `faiss.IndexFlatIP` with L2 normalization
   - Pre-computes 7 nearest-neighbor triplets per DICOM image (~2 hrs on CPU)

3. **Entity alignment** — `scripts/build_entity_cui_map.py`
   - Maps free-text entities to PrimeKG nodes for graph reasoning
   - Stage 1: Exact case-insensitive Neo4j match
   - Stage 2: scispaCy UMLS CUI linking
   - Stage 3: Fuzzy matching (disabled)

4. **PrimeKG subgraph export** — `kg/explore_primekg.py`
   - Focused radiology subgraph for relevant and fast downstream queries
   - 10 seed diagnoses, 2-hop BFS export
   - Graph size: 129,375 nodes, 4,050,249 edges

Coverage: 6/10 diagnoses mapped to PrimeKG. Atelectasis, Consolidation, Lung Opacity, and Pleural Effusion have no PrimeKG disease entity — they exist as phenotypes but lack direct disease linkage.


## Methodology

**Model**: LLaVA-v1.5-7B with LoRA (r=64, α=128, lr=2e-4, 1 epoch, batch=1)

### Prompt Templates

**No-hop (baseline)**:
```
<image>
The image-specific triplets from the knowledge graph are: {kg_triplets}.
And for the given image, {question}
```

**One-hop (KG-augmented)**:
```
<image>
The image-specific triplets from the knowledge graph are: {kg_triplets}.
The multi-hop reasoning chains are: {kg_chains}.
And for the given image, {question} Use the knowledge graph context (if provided) to support your explanation.
```

### Knowledge Graphs

**PrimeKG** — Biomedical knowledge graph with disease-disease and phenotype-phenotype edges.
- Allowed edges: `disease_disease`, `phenotype_phenotype`
- Entities: biological (diseases, symptoms — e.g., fever, cough)
- Coverage: 6/10 diagnoses matched

**RadLex** — Radiology-specific ontology with `May_Cause` edges between radiological findings.
- Entities: radiological signs (e.g., bulging fissure sign, crazy-paving sign)
- More domain-specific: vocabulary directly overlaps with radiologist report language
- Full coverage across all 10 diagnoses

### Chain Format

```
opacity --suggestive_of--> consolidation --phenotype_phenotype--> Pulmonary opacity
```

---

## Evaluation Metrics

### Why Two Metric Families?

NLG metrics measure **surface-level text overlap** — how similar the model's words are to the reference. LLM-as-judge measures **clinical quality** — whether the explanation is medically sound. Both are needed because a model can score well on NLG by repeating common radiology phrases while still missing key findings or reasoning incorrectly.

### NLG Metrics

| Metric | What It Measures | Why We Use It |
|--------|-----------------|---------------|
| **BLEU-1/2/4** | N-gram precision between hypothesis and reference. BLEU-4 uses 4-gram overlap — requires exact phrase matches. | Standard NLG benchmark; BLEU-4 is the paper's reported metric (target: 7.2) |
| **METEOR** | Alignment-based overlap accounting for stemming and synonyms, averaged per sample. | More lenient than BLEU; handles paraphrase better; paper reports this metric (target: 15.1) |
| **ROUGE-L** | Longest Common Subsequence F1 between hypothesis and reference. | Captures fluency and ordering; paper target: 25.0 |
| **CIDEr** | TF-IDF weighted n-gram consensus score across the corpus. Penalizes generic phrases, rewards specific ones. | Strongest signal for medical NLEs — a model saying "opacity" everywhere gets penalised; paper target: 62.2 |
| **Chain Coverage** | % of answers mentioning ≥1 entity from their KG chain. | Measures whether the model actually uses the injected KG context |
| **Avg Hop Depth** | Average number of edges traversed per chain. | Verifies chain structure is correctly 1-hop or 2-hop |

### LLM-as-judge Metrics (Claude Haiku 4.5)

Evaluated on 100 samples (seed=42). Each dimension rated 1–5.

| Dimension | What It Measures |
|-----------|-----------------|
| **Clinical Accuracy** | Are findings factually correct? Does the model avoid hallucinating conditions not in the image? |
| **Completeness** | Does the answer cover the key findings from the reference? (e.g., not stopping at 1 finding when 3 exist) |
| **Reasoning Quality** | Are causal links sound? (e.g., opacity → consolidation → pneumonia — are these justified?) |
| **Language Quality** | Is the explanation clear, concise, and using appropriate medical terminology? |

---

### Update: Judge Robustness on Corrected Triplets

After regenerating corrected test triplets and rerunning LLM-as-judge for `liuhaotian/llava-v1.5-7b`, the RadLex variant remained the strongest system across multiple judge models.

| Judge Model | Base Overall | PrimeKG Overall | RadLex Overall |
|------------|--------------|-----------------|----------------|
| Claude Haiku 4.5 | 2.94 | 2.99 | **3.06** |
| Claude Sonnet 4 | 3.41 | 3.41 | **3.50** |

This matters because it strengthens the claim that the RadLex improvement is not specific to a single LLM judge. The absolute values differ by judge model, but the ranking remains stable: RadLex stays best overall, PrimeKG is mixed, and the baseline remains weakest or tied for weakest.

## Results

### NLG Metrics (n=709)

| Metric | No-hop | One-hop PrimeKG | One-hop RadLex |
|--------|--------|-----------------|----------------|
| BLEU-1 | 32.57 | 36.57 | **38.12** |
| BLEU-2 | 19.93 | 24.46 | **26.22** |
| BLEU-4 | 8.99 | 12.96 | **14.24** |
| METEOR | 30.70 | 35.50 | **37.05** |
| ROUGE-L | 30.71 | 35.23 | **37.69** |
| CIDEr | 59.09 | 90.28 | **107.80** |
| Chain Coverage | 0.0% | 50.07% | **74.19%** |
| Avg Hop Depth | 0.0 | 2.0 | 2.0 |

### LLM-as-judge (n=100, seed=42)

| Dimension | No-hop | One-hop PrimeKG | One-hop RadLex |
|-----------|--------|-----------------|----------------|
| Clinical Accuracy | 2.79 | 2.74 | **2.93** |
| Completeness | 2.08 | 2.15 | **2.29** |
| Reasoning Quality | 3.01 | 2.94 | **3.06** |
| Language Quality | **4.05** | 4.08 | 4.06 |
| **Overall** | 2.98 | 2.98 | **3.08** |

---

## Analysis

### Effect of KG Augmentation

Both PrimeKG and RadLex improve NLG metrics substantially over no-hop. CIDEr is the most sensitive metric — it weights contextually specific phrases heavily, so models that use domain-appropriate terminology score much higher. The jump from no-hop (59.09) to one-hop RadLex (107.80) represents an 82% gain, well above the paper's target of 62.2.

PrimeKG also helps (+53% CIDEr) but trails RadLex on every metric.

### PrimeKG vs RadLex

RadLex outperforms PrimeKG across all NLG metrics. The reason is **ontology fit**: RadLex describes radiological signs (`bulging fissure sign`, `crazy-paving sign`, `ground glass pattern`) whose vocabulary directly overlaps with MIMIC-NLE reference reports written by radiologists. PrimeKG describes biological relationships (`fever`, `cough`, `disease_disease` links) that are clinically meaningful but do not appear in radiology reports.

Chain coverage confirms this — 74% of RadLex answers mention a chain entity vs 50% for PrimeKG. The model cannot use PrimeKG chains for 4 of 10 diagnoses (Atelectasis, Consolidation, Lung Opacity, Pleural Effusion) because they have no PrimeKG disease mapping.

Notably, PrimeKG **hurts clinical accuracy** relative to no-hop (2.74 vs 2.79). Biological entities like `fever` may distract the model into mentioning conditions that are not visually supported by the X-ray.

### NLG vs LLM-judge Gap

The large NLG gains (+82% CIDEr for RadLex) do not translate proportionally to LLM-judge gains (+0.10 overall). This disconnect reveals that the model is **adopting RadLex vocabulary** from the chains but not constructing deeper clinical arguments. Haiku rates answers poorly on completeness (2.29/5) and accuracy (2.93/5) because most answers are still 1–2 sentences that name a finding and attach a diagnosis without connecting multiple findings or explaining the clinical chain of evidence.

### What Needs Improvement

- **Completeness** is the weakest LLM-judge dimension across all three conditions — the model consistently under-reports findings
- **PrimeKG coverage**: 4/10 diagnoses unmatched; RadLex achieves full coverage
- **1-hop depth**: chains that traverse 2+ hops could force longer, more connected explanations
- Results are on the 709-sample demo subset; full dataset validation needed

---

## Dry-run Prompt Examples

**No-hop** (question_id 409):
```
<image>
The image-specific triplets from the knowledge graph are: markings suggestive of edema.
And for the given image, Which signs show that the patient has uncertain Consolidation,
uncertain Edema, positive Lung Opacity, uncertain Pneumonia?
```
→ *"Increased opacity at the right lung base may represent aspiration or infection."*

**One-hop PrimeKG** (same case):
```
<image>
The image-specific triplets from the knowledge graph are: markings suggestive of edema.
The multi-hop reasoning chains are: focal opacity --suggestive_of--> consolidation
--phenotype_phenotype--> Pulmonary opacity.
And for the given image, Which signs show that the patient has uncertain Consolidation,
uncertain Edema, positive Lung Opacity, uncertain Pneumonia?
```
→ *"More focal opacity in the right lower lobe could reflect an area of consolidation."*

The chain adds a clinically linked neighbor (opacity → consolidation) so the model leans toward a specific consolidation explanation rather than a vague opacity description.

---

## File References

| File | Purpose |
|------|---------|
| `scripts/build_demo_llava_json.py` | Builds train/test JSON with KG prompts |
| `scripts/build_entity_cui_map.py` | Maps RadGraph entities to PrimeKG nodes |
| `scripts/build_multihop_chains.py` | Queries Neo4j for 1-hop PrimeKG chains |
| `scripts/build_radlex_chains.py` | Queries RadLex for 1-hop chains |
| `scripts/metrics.py` | NLG metrics: BLEU, METEOR, ROUGE-L, CIDEr |
| `scripts/metrics_llm.py` | LLM-as-judge metrics via Claude Haiku |
| `scripts/modal_demo_train_llava.py` | Fine-tunes LLaVA-v1.5-7B on Modal |
| `scripts/modal_demo_eval_llava.py` | Runs inference on Modal |
| `experiments/demo_answers_no_hop.jsonl` | No-hop model predictions |
| `experiments/demo_answers_one_hop.jsonl` | One-hop PrimeKG predictions |
| `tmp/demo/llava_modal_eval_radlex/demo_answers.jsonl` | One-hop RadLex predictions |
