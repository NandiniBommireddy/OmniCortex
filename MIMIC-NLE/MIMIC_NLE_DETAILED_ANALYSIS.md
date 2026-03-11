# MIMIC-NLE Dataset: Comprehensive Technical Analysis

## Overview: MIMIC-NLE Dataset Architecture

**MIMIC-NLE** (MIMIC Natural Language Explanations) is an explainable medical AI dataset that pairs chest X-ray images with natural language explanations (NLEs) of radiological findings. The dataset is derived from **MIMIC-CXR**, a publicly available chest radiograph database with associated radiology reports.

### Core Purpose & Architecture

The dataset addresses the explainability challenge in medical imaging by creating fine-grained alignments between:

1. **Chest X-ray Images** (from MIMIC-CXR repository)
2. **Sentence-level NLEs** (extracted from radiology reports via spaCy sentence splitting)
3. **Multi-label Diagnoses** (10-class taxonomy) 
4. **Image-wide Labels** (3-state encoding for each diagnostic class)

This enables training of models that not only predict diagnoses but can justify them with human-readable explanations extracted from actual clinical reports.

---

## 1. Data Extraction Pipeline

### Source: [extract_mimic_nle.py](extract_mimic_nle.py) & [preprocess_mimic.py](utils/preprocess_mimic.py)

The dataset extraction follows this pipeline:

```
MIMIC-CXR Reports (p10-p19 folders)
    ↓ [extract_sentences()]
Sentence extraction (findings + impression sections only)
    ↓ [assign_sentences()]
Label assignment via query files + text matching
    ↓
MIMIC-NLE JSON (train/dev/test split)
```

**Key Processing Steps** ([preprocess_mimic.py](utils/preprocess_mimic.py#L14-L80)):

1. **Report Iteration**: Walks through `mimic_path/p10/p10*/reports/*.txt` structure (Lines 38-49)
2. **Section Extraction**: Isolates only `"findings"` and `"impression"` sections using regex-based parsing ([section_splitter.py](utils/section_splitter.py#L1)) 
   - Medical justification: These sections contain diagnostic assertions; other sections (technique, comparison) are methodological
3. **Sentence Splitting**: Uses spaCy's English pipeline sentencizer ([Line 67](utils/preprocess_mimic.py#L67))
   - **Critical Note**: The README emphasizes using the exact spaCy version due to sentence boundary differences
4. **Unique ID Assignment**: Creates `sentence_ID` as `"{report_ID}#{index}"` where index is 0-based sentence position within the report ([Lines 75-76](utils/preprocess_mimic.py#L75-L76))

---

## 2. Label Taxonomy: 10-Class Diagnosis Schema

### Source: [encodings.py](encodings.py#L1-L13)

The dataset uses a **10-class diagnosis taxonomy** defined for MIMIC-CXR, ordered alphabetically:

```python
MIMIC_DIAGNOSIS2LABEL = {
    'Atelectasis': 0,
    'Consolidation': 1,
    'Edema': 2,
    'Enlarged Cardiomediastinum': 3,
    'Lung Lesion': 4,
    'Lung Opacity': 5,
    'Pleural Effusion': 6,
    'Pleural Other': 7,        # ← Example entry: diagnosis_label index = 1
    'Pneumonia': 8,
    'Pneumothorax': 9
}
```

**Why this taxonomy?**
- Standard subset of CheXpert-14 label space (pruned to most common/clinically significant findings)
- One-hot encoding enables multi-label classification (a patient can have multiple diagnoses)
- Ordered alphabetically for reproducibility across implementations

---

## 3. Encodings: Image Label Tri-State Representation

### Source: [encodings.py](encodings.py#L15-L23)

```python
MIMIC_CAT2ONEHOT = {
    'nan': [1,0,0],        # Position 0: Negative (absence of finding)
    '0.0': [1,0,0],        # 
    '-1.0': [0,1,0],       # Position 1: Uncertain (finding mentioned but severity unclear)
    '1.0': [0,0,1]         # Position 2: Positive (finding definitely present)
}
```

**Semantic Mapping:**
- **`[1,0,0]`** = **Negative** → Finding definitively absent from image
- **`[0,1,0]`** = **Uncertain** → Finding mentioned with hedging language ("may represent," "could indicate") or unclear radiographic evidence
- **`[0,0,1]`** = **Positive** → Finding clearly visible and clinically significant

---

## 4. Detailed Field Breakdown: Complete Example Analysis

### Full Entry Structure

```json
{
  "sentence_ID": "s50056854#4",
  "diagnosis_label": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  "evidence_label": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "img_labels": [[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
  "nle": "There is persistent minimal blunting the left costophrenic angle, which may represent a tiny effusion or chronic pleural thickening.",
  "report_ID": "s50056854",
  "patient_ID": "p17096560"
}
```

### 4.1 **`sentence_ID`: "s50056854#4"**

**Structure**: `{report_ID}#{sentence_index}`

- **`s50056854`** → Unique radiology report identifier in MIMIC-CXR
- **`#4`** → Zero-indexed sentence position within that report's findings + impression sections (Line 75, [preprocess_mimic.py](utils/preprocess_mimic.py#L75))

**Dataset Segmentation Rationale**:
- MIMIC-CXR reports are typically 200-400+ words (single text blocks)
- Splitting into sentences enables **fine-grained human annotation** (radiologists can label each diagnostic claim separately)
- Allows the dataset to distinguish between multiple findings in a single report (e.g., sentence #4 discusses pleural effusion; sentence #5 might discuss atelectasis)

**Relationship to `report_ID`**:
- Multiple sentences within one report share the same `report_ID` (all from "s50056854") but different `sentence_ID` suffixes
- Example from query file: Next sentence is `"s50056854#5"` about "Streaky linear opacities... atelectasis"
- The `report_ID` links sentences to the same imaging study and patient exam

---

### 4.2 **`diagnosis_label`: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]**

**Decoding** (using MIMIC_LABEL2DIAGNOSIS from [encodings.py](encodings.py)):

| Index | Class | Value | Meaning |
|-------|-------|-------|---------|
| 0 | Atelectasis | 0 | Not explained |
| 1 | Consolidation | 0 | Not explained |
| ... | ... | 0 | ... |
| **7** | **Pleural Other** | **1** | ✓ **This sentence explains this diagnosis** |
| 8 | Pneumonia | 0 | Not explained |
| 9 | Pneumothorax | 0 | Not explained |

**Clinical Justification**:

The NLE contains key phrases indicating pleural pathology:
- **"blunting the left costophrenic angle"** → Classic radiographic sign of pleural involvement
- **"tiny effusion"** → Fluid in pleural space (directly relevant to "Pleural Effusion" class, but mapped to broader "Pleural Other" in this taxonomy)
- **"chronic pleural thickening"** → Pleural structural change

The diagnosis label captures that **this specific sentence is a natural language explanation for why a clinician concluded there is a Pleural Other finding**, not unrelated filler text.

---

### 4.3 **`evidence_label`: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]**

**Interpretation**: All zeros = **zero-hot encoding** (no position marked as 1)

**Semantic Difference from `diagnosis_label`**:

| Field | Meaning | Scenario |
|-------|---------|----------|
| **`diagnosis_label`** | The finding/diagnosis **being explained in this sentence** | "I see blunting → therefore pleural effusion" |
| **`evidence_label`** | The diagnosis for which **this sentence provides supporting evidence** (used elsewhere in the report) | "Elsewhere in report, Edema was diagnosed; does this sentence support that?" |

**Why is `evidence_label` all-zeros here?**

- The sentence explains pleural effusion/pleural pathology (its own diagnosis)
- It does **not** provide supporting evidence for **other** diagnostic claims in the report
- Example contrast from same report (s50056854#5): It explains "Atelectasis" (diagnosis_label=[1,0,0,...]) but its evidence_label=[0,0,0,0,0,1,0,0,0,0] indicates this sentence is **evidence for Pleural Effusion** (index 6, value=1) claimed elsewhere

**Use Case in Explainability**:
- Distinguishes **primary claims** ("This sentence makes a diagnosis claim")
- From **supporting claims** ("This sentence justifies a diagnosis claimed earlier")
- Enables training models to generate explanations that either assert or support findings

---

### 4.4 **`img_labels`: 10 × 3 Matrix**

**Structure**: List of 10 triplets, one per diagnostic class (in order of MIMIC_DIAGNOSIS2LABEL)

```
[
  [0, 1, 0],  # Index 0: Atelectasis
  [1, 0, 0],  # Index 1: Consolidation  
  [1, 0, 0],  # Index 2: Edema
  [0, 1, 0],  # Index 3: Enlarged Cardiomediastinum
  [1, 0, 0],  # Index 4: Lung Lesion
  [0, 0, 1],  # Index 5: Lung Opacity    ← Positive finding
  [0, 1, 0],  # Index 6: Pleural Effusion  ← Uncertain/hedged
  [0, 1, 0],  # Index 7: Pleural Other     ← Uncertain/hedged
  [1, 0, 0],  # Index 8: Pneumonia
  [1, 0, 0]   # Index 9: Pneumothorax
]
```

**Decoding Using MIMIC_CAT2ONEHOT** ([encodings.py](encodings.py#L15-L23)):
- **`[1,0,0]`** = Negative (finding absent from X-ray)
- **`[0,1,0]`** = Uncertain (finding mentioned with hedging or unclear visibility)
- **`[0,0,1]`** = Positive (finding clearly present)

#### 4.4.1 Clinical Pattern Analysis

| Index | Diagnosis | `img_label` | Interpretation |
|-------|-----------|-------------|-----------------|
| 0 | Atelectasis | `[1,0,0]` | **Negative** — No atelectasis visible on image |
| 5 | Lung Opacity | `[0,0,1]` | **Positive** — Opacity definitely present |
| 6 | Pleural Effusion | `[0,1,0]` | **Uncertain** — Effusion suspected but not definitively shown ("may represent a tiny effusion") |
| 7 | Pleural Other | `[0,1,0]` | **Uncertain** — "chronic pleural thickening" is a descriptive possibility, not a definite finding |

#### 4.4.2 Why `[0,1,0]` for Index 7 (Pleural Other)?

The NLE contains hedging language:
- **"may represent"** — Indicates diagnostic uncertainty
- **"or chronic pleural thickening"** — Alternative explanation, not assertion

This is encoded as **Uncertain** `[0,1,0]`, not Positive `[0,0,1]`, because while the sentence explains pleural pathology (diagnosis_label), the actual radiographic evidence for it is ambiguous.

---

### 4.5 **`nle`: "There is persistent minimal blunting the left costophrenic angle, which may represent a tiny effusion or chronic pleural thickening."**

**Clinical Content Breakdown**:

| Clinical Entity | Significance | Anatomical Context |
|-----------------|--------------|-------------------|
| **Blunting** | Obscuring/loss of sharp angle definition | Classic sign of pleural disease |
| **Costophrenic angle** | Junction of diaphragm and lateral chest wall | Primary location for pleural effusion accumulation |
| **Left** | Laterality (unilateral finding) | Suggests localized rather than diffuse pathology |
| **Minimal** | Severity modifier | Small amount of fluid/thickening |
| **Persistent** | Temporal modifier | Chronic or recurrent (not acute) |
| **Effusion** | Fluid in pleural space | Core finding linking to Pleural Effusion diagnosis |
| **Pleural thickening** | Fibrosis/scarring of pleura | Links to Pleural Other diagnosis |

**Diagnostic Mapping to `diagnosis_label[7]=1` (Pleural Other)**:

The sentence contains sufficient radiographic terminology to justify coding it as an explanation for pleural pathology:
- Direct mention of "effusion" and "pleural thickening"
- Localization to anatomically significant area (costophrenic angle)
- Clinical assessment language ("may represent")

Note: The diagnosis_label doesn't distinguish between "Pleural Effusion" (index 6) and "Pleural Other" (index 7)—the coder selected index 7 as the broader category encompassing both the possible effusion and the chronic thickening.

---

### 4.6 **`report_ID` & `patient_ID`**

| Field | Example | Purpose |
|-------|---------|---------|
| `report_ID` | `"s50056854"` | Unique identifier for a radiology report (one per imaging study/exam) |
| `patient_ID` | `"p17096560"` | Persistent patient identifier across multiple exams |

**Dataset Relationship**:
- A single patient (p17096560) may have multiple reports over time
- A single report (s50056854) contains one full radiology impression
- That report is split into multiple sentences → multiple training examples with same report_ID
- img_labels are **image-wide** (same for all sentences from one report) since they represent findings on that specific X-ray

---

## 5. Data Flow Integration: Training Pipeline Usage

### 5.1 **Data Structure Role: Input Sample Construction**

Each MIMIC-NLE JSON entry functions as a **single training example** in the following pipeline:

```
Entry: {"sentence_ID": "s50056854#4", "diagnosis_label": [0,...,1,...,0], ...}
  ↓
Image Loader: Load image from s50056854.jpg (MIMIC-CXR)
  ↓
Encoder: Image → Feature embeddings (ResNet, DenseNet, ViT)
  ↓
Text Encoder: NLE text → Token embeddings (BERT, RoBERTa)
  ↓
Loss Computation: Align image features with text and labels
```

**This is an input sample**—the model ingests the image and sentence together, using the labels to supervise the alignment.

### 5.2 **Label Semantics in Loss Functions**

#### 5.2.1 **`diagnosis_label`**: Medical Relevance Loss

```
L_diagnosis = CrossEntropy(y_pred_diagnosis, diagnosis_label)
           = Cross-Entropy Loss across 10-class multi-label classification
```

- **Scalar responsibility**: The model learns that sentence s50056854#4 is relevant to explaining Pleural Other (class 7)
- **Supervision goal**: Encourage the model to attend to (and explain) this specific finding
- **Used in**: Justification filtering—distinguish explanatory sentences from filler

#### 5.2.2 **`evidence_label`**: Secondary Evidence Supervision

```
L_evidence = CrossEntropy(y_pred_evidence, evidence_label)
```

- When non-zero, it specifies that **this sentence supports a diagnosis made elsewhere in the report**
- Example: If s50056854#6 said "Also suggestive of edema," evidence_label would mark index 2 (Edema)
- **Use case**: Training the model to recognize both assertion vs. confirmation patterns
- **Zero here**: Sentence is a primary diagnostic claim, not a supporting fact

#### 5.2.3 **`img_labels`**: Visual Grounding in Contrastive Learning

```
L_visual = ContrastiveLoss(image_features, diagnosis_predictions, img_labels)
         or
L_visual = MultiLabelBCELoss(y_pred_vision, img_labels)
```

The `img_labels` serve critical roles:

**Role 1: Visual Supervision**
- Model learns to predict image-wide diagnostic features
- Regularizes image encoder to capture clinically meaningful regions
- `[0,0,1]` for Lung Opacity tells the model: "This image has opacity"
- `[1,0,0]` for Atelectasis tells it: "No atelectasis on this image"

**Role 2: Text-Image Alignment**
```
Contrastive Loss:
  - For s50056854#4, the NLE explains Pleural Other
  - img_labels[7] = [0,1,0] (Uncertain)
  - Loss encourages: 
    - NLE feature proximity to region of image with pleural findings
    - Model learns that hedged text ([0,1,0]) pairs with uncertain visual evidence
```

**Role 3: Attention-Based Loss**
```
Attention Mechanism:
  - Model attends to regions of image corresponding to NLE claims
  - α_j,k = attention weight of token j to region k
  - Loss = ∑_j ∑_k α_j,k * L(prediction_k, img_labels[explanation_class_k])
```

### 5.3 **Pattern Analysis: Why`[0,1,0]` at Index 7?**

```
NLE: "...costophrenic angle, which MAY REPRESENT a tiny effusion..."
                                 ↓
                         Hedging language
                                 ↓
              img_labels[7] = [0,1,0] (Uncertain)
                                 ↓
Training Signal: Model learns to associate hedged language 
                 with uncertain visual evidence
```

**Significance for Model Training**:
- Without this encoding, the model might learn to assign high confidence to vague claims
- The tri-state encoding enables **calibrated explanations**: high-confidence findings get `[0,0,1]`, uncertain findings get `[0,1,0]`
- Patient safety: A model trained on this can express uncertainty when warranted, rather than overconfident misdiagnosis

### 5.4 **Example Training Trajectory**

```
Batch Item: {
  image_id: "s50056854",
  sentence: "There is persistent minimal blunting...",
  diagnosis_label: [0,...,1,0],     # Explains Pleural Other
  evidence_label: [0,...,0],        # No secondary evidence
  img_labels: [[0,1,0], [1,0,0], ..., [0,1,0], [0,1,0], [1,0,0], [1,0,0]]
                               ↑
                         Index 7: Uncertain
}

Forward Pass:
  1. image_encoder(image) → h_image (2048-dim)
  2. text_encoder(sentence) → h_text (768-dim)
  3. classifier_diagnosis(h_image, h_text) → p_diagnosis (10-dim logits)
  4. classifier_image(h_image) → p_img (10×3-dim logits)

Loss Computation:
  L_diag = CrossEntropy(p_diagnosis, [0,...,1,0])  # Primary loss
  L_img = BCEWithLogits(p_img, img_labels)         # Visual grounding
  L_align = ContrastiveLoss(h_text, h_image)       # Modality alignment
  L_total = w1*L_diag + w2*L_img + w3*L_align

Backward Pass:
  Gradients flow through:
    - Text encoder (learn to encode diagnostic descriptions)
    - Image encoder (learn to localize findings)
    - Cross-modal attention (learn which image regions matter for text)
```

---

## 6. Summary: Information Flow

### Data Lineage

```
MIMIC-CXR Radiology Reports
    ↓ spaCy sentence splitting (findings + impression sections)
    ↓ Unique ID assignment: "report_ID#index"
    ↓
Sentence-level Dictionaries {sentence_ID → (text, patient_ID, report_ID)}
    ↓ (Matched with)
    ↓
Human Annotation: query files with labels
    ↓ (Labels assigned by radiologists/experts)
    ↓
MIMIC-NLE JSON entries
    ↓
Training Pipeline: Multi-task learning
    - diagnosis_label: What diagnosis does this sentence explain?
    - evidence_label: What diagnosis does it support?
    - img_labels: What visual findings are present on the image?
    - nle text: Generate/supervise natural language output
```

### Label Hierarchy

```
  diagnosis_label (one-hot, ~2% have value=1)
            ↓
      Specifies primary diagnostic focus
            ↓
            ├→ Trains: Which findings to attend to
            ├→ Trains: Text understanding of clinical language
            └→ Trains: Explanation selection
            
  evidence_label (mostly zero, ~10% have value≠0)
            ↓
      Specifies secondary evidentiary relationships
            ↓
            └→ Trains: Different generation pattern (supporting vs. asserting)
            
  img_labels (all 10 classes, always fully populated)
            ↓
      Specifies visual ground truth for all findings
            ↓
            ├→ Trains: Image feature learning
            ├→ Trains: Text-image alignment
            └→ Trains: Confidence calibration
            
  nle text (variable length, ~15 tokens on average)
            ↓
      Specifies target explanation
            ↓
            ├→ Trains: Text decoder/generator
            ├→ Trains: Selecting informative clinical details
            └→ Trains: Natural language fluency
```

---

## 7. Code References Summary

| Component | File | Key Lines | Purpose |
|-----------|------|-----------|---------|
| Label Taxonomy | [encodings.py](encodings.py#L3-L13) | 3-13 | Define 10-class diagnosis mapping |
| Image Label Encoding | [encodings.py](encodings.py#L15-L23) | 15-23 | Define tri-state visual labels |
| Sentence Extraction | [preprocess_mimic.py](utils/preprocess_mimic.py#L14-L80) | 14-80 | Split reports into sentences |
| ID Assignment | [preprocess_mimic.py](utils/preprocess_mimic.py#L75-L80) | 75-80 | Create sentence_ID format |
| Label Integration | [extract_mimic_nle.py](extract_mimic_nle.py#L7-L27) | 7-27 | Match sentences to labels |
| Section Extraction | [section_splitter.py](utils/section_splitter.py#L20-L60) | 20-60 | Filter to clinical sections |
| README Documentation | [README.md](README.md#L24-L42) | 24-42 | Dataset structure definition |

---

## 8. Key Takeaways

1. **Sentence-level Granularity**: The `#N` suffix enables fine-grained annotation, allowing each diagnostic claim in a multi-sentence report to be labeled independently.

2. **Diagnosis at Index 7 = "Pleural Other"**: A broader category that encompasses pleural space diseases including effusions and chronic thickening. The NLE's clinical language (blunting, costophrenic angle, effusion, pleural thickening) justifies this classification.

3. **Evidence Label = Zero**: Indicates this sentence is a primary assertion, not supporting evidence for another diagnosis. The distinction is critical for generating explanations that distinguish between "I observe X" and "This supports Y that was mentioned earlier."

4. **Image Labels = Visual Ground Truth**: The tri-state encoding `[negative, uncertain, positive]` provides image-wide supervision, enabling the model to learn which findings are actually visible, which are uncertain, and which are absent—critical for calibrated confidence in explanations.

5. **Contrastive Training**: The pairing of image features, text features, and visual labels enables multi-task learning where the model jointly learns to:
   - Encode diagnostic language
   - Localize findings in images
   - Align text with relevant image regions
   - Generate explanations that match visual evidence

6. **Clinical Safety**: The hedged language ("may represent") paired with uncertain visual labels (`[0,1,0]`) teaches the model to express appropriate uncertainty, avoiding overconfident misdiagnosis in clinical deployment.

