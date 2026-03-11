# MIMIC-NLE: Quick Reference Visual Guide

## 1-Minute Data Structure Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  MIMIC-NLE JSON Entry (Training Example)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  sentence_ID:      "s50056854#4"                                    │
│                     └─┬─ Report ID    ┬─ Sentence index (0-based)   │
│                       └─ Comes from:  └─ From sentence splitting    │
│                          MIMIC-CXR       of findings+impression      │
│                                                                      │
│  diagnosis_label:  [0,0,0,0,0,0,0,1,0,0]                           │
│                                       └─ Index 7 = "Pleural Other"  │
│                                                                      │
│  evidence_label:   [0,0,0,0,0,0,0,0,0,0]  ← All zeros (primary    │
│                                              claim, not support)     │
│                                                                      │
│  img_labels:       [[0,1,0], [1,0,0], ..., [0,1,0], [0,1,0], ...]  │
│                      └─┬─ Uncertain         └─ For each of 10      │
│                        └─ [neg, unc, pos]     diagnostic classes    │
│                                                                      │
│  nle:              "There is persistent minimal blunting the left   │
│                     costophrenic angle, which may represent a tiny  │
│                     effusion or chronic pleural thickening."        │
│                     └─ Clinical entities: costophrenic angle,       │
│                        effusion, pleural thickening                 │
│                                                                      │
│  report_ID:        "s50056854"  ← Groups 6 sentences from same exam │
│  patient_ID:       "p17096560"  ← Patient across multiple exams     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Label Encoding Reference

### Diagnosis/Evidence Labels (One-hot, 10 dimensions)

```
Index    Diagnosis Class              Used by
────────────────────────────────────────────────
  0      Atelectasis                  diagnosis_label OR evidence_label
  1      Consolidation                
  2      Edema                        
  3      Enlarged Cardiomediastinum   
  4      Lung Lesion                  
  5      Lung Opacity                 
  6      Pleural Effusion             
  7 ◄────Pleural Other                ◄ EXAMPLE HAS VALUE=1
  8      Pneumonia                    
  9      Pneumothorax                 

Zero vectors ([0,0,...,0]) = No assignment
One-hot vectors = Single diagnosis assigned
```

### Image Labels (10 × 3 Matrix)

```
For each of the 10 diagnostic classes, img_labels provides:

   [1, 0, 0]  →  NEGATIVE      (finding absent from this image)
   [0, 1, 0]  →  UNCERTAIN     (finding mentioned with hedging/unclear visibility)
   [0, 0, 1]  →  POSITIVE      (finding clearly present on image)

Example row for index 7 (Pleural Other):
   [0, 1, 0]  →  Uncertain (NLE says "may represent" = hedging)
```

---

## 3. Semantic Difference: diagnosis_label vs. evidence_label

```
┌──────────────────────────────────────────────────────────────────┐
│ diagnosis_label = "What does THIS sentence explain?"            │
│                   (Direct claim by sentence)                     │
│                                                                  │
│ evidence_label  = "What other diagnosis does THIS sentence      │
│                    support?" (Mentioned elsewhere in report)     │
│                                                                  │
│ EXAMPLE:                                                         │
│ ───────                                                          │
│                                                                  │
│ Sentence #4: "...blunting costophrenic angle..."               │
│   diagnosis_label[7] = 1  ←  Explains Pleural Other             │
│   evidence_label[*]  = 0  ←  No secondary support role          │
│                                                                  │
│ Sentence #5: "...atelectasis or scarring..."                   │
│   diagnosis_label[0] = 1  ←  Explains Atelectasis              │
│   evidence_label[6] = 1  ←  ALSO supports Pleural Effusion      │
│                            (found in broader report)            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Clinical-to-Diagnostic Mapping

```
NLE: "...blunting the left costophrenic angle, 
      which may represent a tiny effusion or 
      chronic pleural thickening..."

Radiological Signs          Clinical Significance       Maps to
──────────────────────────────────────────────────────────────────
Blunting                     Loss of angle sharpness    } 
Costophrenic angle           Site of fluid accumulation } → Pleural
Effusion (tiny)              Fluid in pleural space     }    Other
Pleural thickening           Fibrosis/scarring          }    (Index 7)

Language Cue                 Interpretation
──────────────────────────────────────────
"minimal"                    Small amount
"persistent"                 Chronic/recurrent
"may represent"              Diagnostic uncertainty
"or"                         Alternative explanation
                             ─────────────────
                    ↓ Encodes as [0,1,0] (Uncertain)
```

---

## 5. Training Data Flow

```
        MIMIC-CXR X-ray Image
        ↓ + Associated Text from s50056854
┌───────────────────────────────────────────┐
│   Image Encoder → h_image (2048-dim)      │
│   Text Encoder  → h_text (768-dim)        │
└───────┬─────────────────────────────────┬─┘
        ↓                                 ↓

┌─────────────────────┐   ┌─────────────────────────┐
│  Diagnosis Pred.    │   │  Image-wide Pred.       │
│  logits (10-dim)    │   │  logits (10×3-dim)      │
├─────────────────────┤   ├─────────────────────────┤
│ P(explain each dx)  │   │ P(neg|unc|pos per dx)  │
│                     │   │                         │
│ Supervised by:      │   │ Supervised by:          │
│ diagnosis_label     │   │ img_labels              │
│ evidence_label      │   │                         │
└─────────────────────┘   └─────────────────────────┘
        ↓                                 ↓
        └────────────┬──────────────────┘
                     ↓
        Multi-task Loss Function
        
L_total = w₁·L_diagnosis + 
          w₂·L_evidence + 
          w₃·L_image + 
          w₄·L_alignment
```

---

## 6. Key Code Locations

```
Concept                    File                        Location
────────────────────────────────────────────────────────────────
10-class taxonomy          encodings.py               Lines 3-13
[neg,unc,pos] encoding    encodings.py               Lines 15-23
Sentence extraction       preprocess_mimic.py        Lines 14-80
ID assignment: "#4"       preprocess_mimic.py        Lines 75-80
Report section filtering  section_splitter.py        Lines 20-60
Label attachment          extract_mimic_nle.py       Lines 7-27
Dataset spec              README.md                  Lines 24-42
```

---

## 7. The `#4` Suffix Deep Dive

```
Report s50056854.txt (original MIMIC-CXR file):
┌─────────────────────────────────────────────────┐
│ IMPRESSION:                                     │
│ Finding 0: "Small opacities left lung base..."  │
│ Finding 1: "Cardiac silhouette normal..."       │
│ ...                                             │
│ Finding 4: ← THIS SENTENCE                      │
│   "There is persistent minimal blunting        │
│    the left costophrenic angle, which may      │
│    represent a tiny effusion or chronic        │
│    pleural thickening."                        │
│ ...                                             │
└─────────────────────────────────────────────────┘
                    ↓ (spaCy splits into sentences)
                    ↓
         sentence_ID = "s50056854#4"
         (4th sentence, 0-indexed)

Rationale:
──────────
• Report text is 200-400+ words → needs segmentation
• spaCy sentence splitter creates boundaries
• Fine-grained annotation: each sentence gets its own labels
• Multiple diagnoses in single report are separated
```

---

## 8. The `[0,1,0]` Pattern in img_labels[7]

```
Why uncertain for Pleural Other (index 7)?

"...may represent a tiny effusion or 
  chronic pleural thickening..."
          ↓
    Hedging language
          ↓
   Not definitive diagnosis
          ↓
    Expert radiologist codes:
    
    img_labels[7] = [0, 1, 0]  (Uncertain)
    NOT [0, 0, 1] (Positive)
    
Contrast with img_labels[5] (Lung Opacity):
    
    [0, 0, 1]  (Positive)
    ↑
  Why? "Opacities" are more clearly delineated
       on X-ray; not hedged in NLE
```

---

## 9. Multi-Label Nature Example

```
From same report (s50056854):

Sentence #4: diagnosis=[0,0,0,0,0,0,0,1,0,0]  (Pleural Other only)
Sentence #5: diagnosis=[1,0,0,0,0,0,0,0,0,0]  (Atelectasis only)

Same image, DIFFERENT sentences highlight DIFFERENT findings:
Sentence #4 extracts: Pleural pathology
Sentence #5 extracts: Atelectasis/scarring

BUT img_labels stay SAME for both:
  (Because image findings are fixed across all sentences
   from one report—they describe the SAME X-ray)

This enables the model to learn:
  • Which image regions correspond to which clinical descriptions
  • How to select relevant sentences for each finding
  • Text-image alignment for explainability
```

---

## 10. Why This Dataset Matters for Explainable Medical AI

```
Challenge: Medical AI often lacks interpretability
          ("Black box" diagnosis without justification)

Solution: MIMIC-NLE provides
┌─────────────────────────────────────────────┐
│ • Sentence-level alignment                  │
│   (Know WHICH SENTENCE explains WHICH finding) │
│                                             │
│ • Clinical language grounding               │
│   (Learn actual radiological terminology)   │
│                                             │
│ • Visual-semantic alignment                 │
│   (Link image regions to clinical language) │
│                                             │
│ • Calibrated uncertainty                    │
│   (Express "uncertain" when warranted)      │
│                                             │
│ • Human-readable output                     │
│   (Generate explanations that patients can  │
│    understand, not just predictions)        │
└─────────────────────────────────────────────┘

Training Result:
  Model learns to:
  ✓ Predict diagnoses (diagnosis_label)
  ✓ Explain them in human language (nle)
  ✓ Ground explanations in image regions (img_labels + attention)
  ✓ Express appropriate uncertainty (tri-state encoding)
```

