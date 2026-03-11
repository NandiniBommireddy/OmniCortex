# MIMIC-NLE: Code-Based Implementation Reference

## 1. Label Encoding Implementation

### From [encodings.py](encodings.py)

```python
# DIAGNOSIS-LABEL MAPPING (from encodings.py, lines 3-13)
MIMIC_DIAGNOSIS2LABEL = {
    'Atelectasis': 0,
    'Consolidation': 1,
    'Edema': 2,
    'Enlarged Cardiomediastinum': 3,
    'Lung Lesion': 4,
    'Lung Opacity': 5,
    'Pleural Effusion': 6,
    'Pleural Other': 7,        # ← Example diagnosis_label[7]=1
    'Pneumonia': 8,
    'Pneumothorax': 9
}

# Inverse mapping for decoding
MIMIC_LABEL2DIAGNOSIS = {v: k for k, v in MIMIC_DIAGNOSIS2LABEL.items()}

# IMAGE LABEL ENCODING (from encodings.py, lines 15-23)
MIMIC_CAT2ONEHOT = {
    'nan': [1,0,0],        # Negative
    '0.0': [1,0,0],        # Maps to "Negative"
    '-1.0': [0,1,0],       # Maps to "Uncertain"
    '1.0': [0,0,1]         # Maps to "Positive"
}

# Alternative naming convention (more intuitive)
MIMIC_STR2ONEHOT = {
    'nan': [1,0,0],
    'Negative': [1,0,0],
    'Uncertain': [0,1,0],
    'Positive': [0,0,1]
}
```

### Decoding Example

```python
def decode_entry(entry):
    """Decode a MIMIC-NLE JSON entry to human-readable format"""
    
    # Decode diagnosis_label
    diagnosis_index = entry['diagnosis_label'].index(1) \
        if 1 in entry['diagnosis_label'] else None
    diagnosis_name = MIMIC_LABEL2DIAGNOSIS.get(diagnosis_index) \
        if diagnosis_index is not None else "None"
    
    # Decode evidence_label
    evidence_index = entry['evidence_label'].index(1) \
        if 1 in entry['evidence_label'] else None
    evidence_name = MIMIC_LABEL2DIAGNOSIS.get(evidence_index) \
        if evidence_index is not None else "None"
    
    # Decode img_labels for each class
    image_findings = {}
    for class_idx, class_name in MIMIC_LABEL2DIAGNOSIS.items():
        img_label = entry['img_labels'][class_idx]
        if img_label == [1, 0, 0]:
            status = "Negative"
        elif img_label == [0, 1, 0]:
            status = "Uncertain"
        elif img_label == [0, 0, 1]:
            status = "Positive"
        else:
            status = "Invalid"
        image_findings[class_name] = status
    
    return {
        'sentence_ID': entry['sentence_ID'],
        'diagnosis_explained': diagnosis_name,
        'evidence_for': evidence_name,
        'nle': entry['nle'],
        'image_wide_findings': image_findings,
        'report_ID': entry['report_ID'],
        'patient_ID': entry['patient_ID']
    }

# Apply to example entry
example = {
    "sentence_ID": "s50056854#4",
    "diagnosis_label": [0,0,0,0,0,0,0,1,0,0],
    "evidence_label": [0,0,0,0,0,0,0,0,0,0],
    "img_labels": [[0,1,0], [1,0,0], [1,0,0], [0,1,0], 
                   [1,0,0], [0,0,1], [0,1,0], [0,1,0], 
                   [1,0,0], [1,0,0]],
    "nle": "There is persistent minimal blunting the left...",
    "report_ID": "s50056854",
    "patient_ID": "p17096560"
}

decoded = decode_entry(example)
print(f"Diagnosis Explained: {decoded['diagnosis_explained']}")
# Output: Diagnosis Explained: Pleural Other

print(f"Image-wide Findings (first 3):")
for dx, status in list(decoded['image_wide_findings'].items())[:3]:
    print(f"  {dx}: {status}")
# Output:
#   Atelectasis: Negative
#   Consolidation: Negative
#   Edema: Negative
# ... (note: Lung Opacity is Positive, Pleural Other is Uncertain)
```

---

## 2. Sentence Extraction: Implementation Details

### From [preprocess_mimic.py](utils/preprocess_mimic.py) Lines 14-80

```python
import os
import math
from spacy.lang.en import English

def extract_sentences(mimic_path, report_count=math.inf, patients=None):
    """
    Extract sentences from MIMIC-CXR reports.
    
    Key processing steps:
    1. Iterate through p10-p19 folders
    2. For each patient/report, extract findings + impression sections
    3. Split into sentences using spaCy
    4. Create sentence_ID as "{report_ID}#{sentence_index}"
    
    Args:
        mimic_path: Path to MIMIC-CXR directory (p10-p19 structure)
        report_count: Upper limit on number of reports to process
        patients: Optional list of patient IDs to process (None = all)
    
    Returns:
        dict: Mapping of sentence_ID → sentence metadata
    """
    
    output = {}
    nlp = English()
    sentencizer = nlp.add_pipe("sentencizer")
    
    # Iterate folder structure: p10-p19 → patient_IDs → reports
    for subfolder in os.listdir(mimic_path):
        subfolder_path = os.path.join(mimic_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
            
        for patient_ID in os.listdir(subfolder_path):
            if patients and patient_ID not in patients:
                continue
            patient_path = os.path.join(subfolder_path, patient_ID)
            
            if not os.path.isdir(patient_path):
                continue
                
            for report_filename in os.listdir(patient_path):
                report_path = os.path.join(patient_path, report_filename)
                if not os.path.isfile(report_path):
                    continue
                
                # Read report
                with open(report_path, "r") as file:
                    report = file.read()
                
                # Extract sections (from section_splitter.py)
                sections, section_names, _ = section_text(report)
                
                # CRITICAL STEP: Keep only 'findings' and 'impression' sections
                # (These contain diagnostic claims; other sections are metadata)
                relevant_sections = [
                    text.replace("\n", "")
                    for text, name in zip(sections, section_names)
                    if name in ["impression", "findings"]
                ]
                
                if len(relevant_sections) == 0:
                    continue
                
                # Concatenate relevant sections
                report_text = "".join(relevant_sections)
                report_text = report_text.replace("  ", " ")
                
                # SENTENCE SPLITTING (spaCy)
                # This creates the #N suffix in sentence_ID
                doc = nlp(report_text)
                sentences = [str(sent).lstrip(" ") for sent in doc.sents]
                
                # Create entry for each sentence
                for sentence_index, sentence_text in enumerate(sentences):
                    # Extract report_ID from filename (remove .txt)
                    report_ID = report_filename.replace(".txt", "")
                    
                    # THIS CREATES THE KEY: "report_ID#index"
                    sentence_ID = f"{report_ID}#{sentence_index}"
                    
                    entry = {
                        "sentence_ID": sentence_ID,
                        "patient_ID": patient_ID,
                        "report_ID": report_ID,
                        "sentence": sentence_text
                    }
                    
                    output[sentence_ID] = entry
    
    return output


# EXAMPLE OUTPUT (for one report with 6 sentences):
# output = {
#     "s50056854#0": {
#         "sentence_ID": "s50056854#0",
#         "patient_ID": "p17096560",
#         "report_ID": "s50056854",
#         "sentence": "First sentence from findings/impression..."
#     },
#     "s50056854#1": { ... },
#     "s50056854#2": { ... },
#     "s50056854#3": { ... },
#     "s50056854#4": {  # ← OUR EXAMPLE
#         "sentence_ID": "s50056854#4",
#         "patient_ID": "p17096560",
#         "report_ID": "s50056854",
#         "sentence": "There is persistent minimal blunting the left..."
#     },
#     "s50056854#5": { ... }
# }
```

---

## 3. Label Assignment: From Query Files to Final Dataset

### From [extract_mimic_nle.py](extract_mimic_nle.py)

```python
import json
from utils.json_processing import read_jsonl_lines, write_jsonl_lines
from utils.preprocess_mimic import extract_sentences

def assign_sentences(query_file, data):
    """
    Match sentence IDs in query files to extracted sentences.
    
    Query files (created by radiologists) contain:
        {
            "sentence_ID": "s50056854#4",
            "diagnosis_label": [0,0,0,0,0,0,0,1,0,0],
            "evidence_label": [0,0,0,0,0,0,0,0,0,0],
            "img_labels": [[0,1,0], ..., [1,0,0]]
        }
    
    This function adds the actual NLE text and metadata:
        {
            "sentence_ID": "s50056854#4",
            "nle": "There is persistent minimal...",
            "report_ID": "s50056854",
            "patient_ID": "p17096560",
            "diagnosis_label": [...],
            "evidence_label": [...],
            "img_labels": [...]
        }
    """
    
    nles = []
    
    for query_entry in query_file:
        # Lookup the extracted sentence by ID
        sentence_ID = query_entry["sentence_ID"]
        source_data = data.get(sentence_ID)
        
        if source_data is None:
            # Handle missing sentence (sentence not extracted)
            continue
        
        # Copy all query labels
        nle_entry = query_entry.copy()
        
        # ADD the actual NLE text from extraction
        nle_entry["nle"] = source_data["sentence"]
        nle_entry["report_ID"] = source_data["report_ID"]
        nle_entry["patient_ID"] = source_data["patient_ID"]
        
        nles.append(nle_entry)
    
    return nles


# MAIN PROCESSING FUNCTION
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_path", type=str, 
                       help="path to MIMIC-CXR reports (p10-p19 structure)")
    args = parser.parse_args()
    
    # STEP 1: Extract all sentences from MIMIC-CXR
    print("Extracting sentences from MIMIC-CXR...")
    sentence_data = extract_sentences(args.reports_path)
    # Returns: {"s50056854#4": {"sentence_ID": "...", "sentence": "..."}, ...}
    
    # STEP 2: Process each dataset split (train/dev/test)
    for split in ["dev", "test", "train"]:
        print(f"Processing {split} split...")
        
        # Load radiologist annotations
        query_file = read_jsonl_lines(f"mimic-nle/query/{split}-query.json")
        # Format: [{"sentence_ID": "s50056854#4", "diagnosis_label": [...], ...}]
        
        # Assign (match queries to extracted sentences)
        nles = assign_sentences(query_file, sentence_data)
        
        # Write final dataset
        write_jsonl_lines(f"mimic-nle/mimic-nle-{split}.json", nles)
        # Output: {"sentence_ID": "...", "nle": "...", "diagnosis_label": [...], ...}
    
    print("Done! MIMIC-NLE train/dev/test created successfully.")
```

---

## 4. Data Loader Pseudo-Code for Training

```python
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, ResNet50

class MIMICNLEDataset(Dataset):
    """PyTorch Dataset for MIMIC-NLE training"""
    
    def __init__(self, json_file, image_dir, max_tokens=128):
        # Load JSONL file
        with open(json_file) as f:
            self.entries = [json.loads(line) for line in f]
        
        self.image_dir = image_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_tokens = max_tokens
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        report_ID = entry['report_ID']
        
        # Load image
        image_path = f"{self.image_dir}/{report_ID}.jpg"
        image = load_image(image_path)  # → tensor (3, 224, 224)
        
        # Tokenize NLE
        tokens = self.tokenizer.encode_plus(
            entry['nle'],
            max_length=self.max_tokens,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Labels
        diagnosis_label = torch.tensor(entry['diagnosis_label'], 
                                       dtype=torch.float)
        evidence_label = torch.tensor(entry['evidence_label'], 
                                      dtype=torch.float)
        img_labels = torch.tensor(entry['img_labels'], 
                                  dtype=torch.float)  # (10, 3)
        
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'diagnosis_label': diagnosis_label,        # (10,)
            'evidence_label': evidence_label,          # (10,)
            'img_labels': img_labels,                  # (10, 3)
            'sentence_ID': entry['sentence_ID'],
            'nle_text': entry['nle']
        }


# Training setup
dataset = MIMICNLEDataset('mimic-nle/mimic-nle-train.json',
                          'path/to/MIMIC-CXR/images')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example batch structure:
# batch = {
#     'image': (32, 3, 224, 224),
#     'input_ids': (32, 128),
#     'diagnosis_label': (32, 10),      ← One-hot: most will be [0,...,0,1,0,...]
#     'evidence_label': (32, 10),       ← Many all-zeros, some non-zero
#     'img_labels': (32, 10, 3),        ← All filled: visual ground truth
#     'sentence_ID': list of 32 strings
#     'nle_text': list of 32 strings
# }
```

---

## 5. Loss Function Implementation

```python
import torch
import torch.nn as nn

class MIMICNLELoss(nn.Module):
    """Multi-task loss for MIMIC-NLE training"""
    
    def __init__(self, weights={'diagnosis': 1.0, 'evidence': 0.5, 'image': 1.0}):
        super().__init__()
        self.weights = weights
        
        # Loss functions
        self.diagnosis_loss = nn.BCEWithLogitsLoss()
        self.evidence_loss = nn.BCEWithLogitsLoss()
        self.image_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: {
                'diagnosis': (batch, 10),      # Logits for diagnosis prediction
                'evidence': (batch, 10),       # Logits for evidence prediction
                'image': (batch, 10, 3)        # Logits for image label prediction
            }
            targets: {
                'diagnosis_label': (batch, 10),    # Ground truth diagnosis
                'evidence_label': (batch, 10),     # Ground truth evidence
                'img_labels': (batch, 10, 3)       # Ground truth image labels
            }
        """
        
        # 1. DIAGNOSIS LOSS
        # ─────────────────
        # Teaches model which diagnosis THIS sentence explains
        # Example: diagnosis_label = [0,0,0,0,0,0,0,1,0,0]
        #          → Model learns this sentence explains class 7
        L_diagnosis = self.diagnosis_loss(
            predictions['diagnosis'],
            targets['diagnosis_label']
        )
        
        # 2. EVIDENCE LOSS
        # ────────────────
        # Teaches model which OTHER diagnosis this sentence supports
        # Example: evidence_label = [0,0,0,0,0,0,0,0,0,0]
        #          → This sentence provides no evidence for other diagnoses
        L_evidence = self.evidence_loss(
            predictions['evidence'],
            targets['evidence_label']
        )
        
        # 3. IMAGE LOSS
        # ─────────────
        # Teaches model to predict image-wide findings
        # Reshape predictions for BCEWithLogitsLoss
        # predictions['image']: (batch, 10, 3)
        # targets['img_labels']: (batch, 10, 3)
        L_image = self.image_loss(
            predictions['image'],  # Logits for [neg, uncertain, positive]
            targets['img_labels']  # Soft targets or one-hot
        )
        
        # Weighted combination
        L_total = (self.weights['diagnosis'] * L_diagnosis +
                  self.weights['evidence'] * L_evidence +
                  self.weights['image'] * L_image)
        
        return {
            'total': L_total,
            'diagnosis': L_diagnosis.item(),
            'evidence': L_evidence.item(),
            'image': L_image.item()
        }


# Example usage in training loop
loss_fn = MIMICNLELoss(weights={'diagnosis': 1.0, 'evidence': 0.5, 'image': 1.0})

for batch in dataloader:
    # Forward pass
    predictions = model(batch['image'], batch['input_ids'], batch['attention_mask'])
    
    # Compute loss
    loss_dict = loss_fn(predictions, {
        'diagnosis_label': batch['diagnosis_label'],
        'evidence_label': batch['evidence_label'],
        'img_labels': batch['img_labels']
    })
    
    # Backward pass
    loss_dict['total'].backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Logging
    print(f"Loss: {loss_dict['total']:.4f} " +
          f"(diag={loss_dict['diagnosis']:.4f}, " +
          f"ev={loss_dict['evidence']:.4f}, " +
          f"img={loss_dict['image']:.4f})")
```

---

## 6. Model Architecture Sketch

```python
class MIMICNLEModel(nn.Module):
    """Simplified architecture for MIMIC-NLE"""
    
    def __init__(self):
        super().__init__()
        
        # Image encoder: ResNet-50 or DenseNet
        self.image_encoder = torchvision.models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(2048, 2048)  # Feature dim
        
        # Text encoder: BERT
        from transformers import BertModel
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # Projection layers (align image and text to shared space)
        self.image_proj = nn.Linear(2048, 512)
        self.text_proj = nn.Linear(768, 512)
        
        # Task-specific heads
        self.diagnosis_head = nn.Linear(512 + 768, 10)      # (img+text)→diagnosis
        self.evidence_head = nn.Linear(512 + 768, 10)       # (img+text)→evidence
        self.image_head = nn.Linear(2048, 10 * 3)           # img→10 classes × 3-state
    
    def forward(self, image, input_ids, attention_mask):
        # Image encoding
        h_image = self.image_encoder(image)  # (batch, 2048)
        h_image_proj = self.image_proj(h_image)  # (batch, 512)
        
        # Text encoding
        text_output = self.text_encoder(input_ids, 
                                        attention_mask=attention_mask)
        h_text = text_output.pooler_output  # (batch, 768)
        h_text_proj = self.text_proj(h_text)  # (batch, 512)
        
        # Fused representation
        h_fused = torch.cat([h_image_proj, h_text], dim=1)  # (batch, 512+768)
        
        # Task predictions
        diagnosis_logits = self.diagnosis_head(h_fused)  # (batch, 10)
        evidence_logits = self.evidence_head(h_fused)  # (batch, 10)
        image_logits = self.image_head(h_image)  # (batch, 10*3) → reshape to (batch, 10, 3)
        
        return {
            'diagnosis': diagnosis_logits,
            'evidence': evidence_logits,
            'image': image_logits.reshape(-1, 10, 3)
        }
```

---

## 7. Inference Example: Generating Explanations

```python
def generate_explanation(model, image_path, device='cuda'):
    """
    Inference example: predict diagnosis and generate/select NLE
    """
    
    # Load image
    image = load_image(image_path)
    image = image.to(device)
    
    # Forward pass (image only, for diagnosis prediction)
    with torch.no_grad():
        h_image = model.image_encoder(image.unsqueeze(0))  # (1, 2048)
        
        # Predict image-wide findings
        image_logits = model.image_head(h_image)  # (1, 10, 3)
        image_probs = torch.softmax(image_logits, dim=-1)  # (1, 10, 3)
        
        # Decode findings
        findings = {}
        for class_idx, class_name in enumerate(['Atelectasis', 'Consolidation', ...]):
            state_probs = image_probs[0, class_idx]  # (3,): [neg, unc, pos] probabilities
            
            if state_probs[2] > 0.5:  # Positive
                findings[class_name] = "Positive"
            elif state_probs[1] > 0.3:  # Uncertain
                findings[class_name] = "Uncertain"
            else:  # Negative
                findings[class_name] = "Negative"
    
    print("Predicted Image-wide Findings:")
    for finding, status in findings.items():
        print(f"  {finding}: {status}")
    
    # Step 2: RETRIEVE or GENERATE explanations for positive findings
    positive_findings = [f for f, status in findings.items() if status == "Positive"]
    
    print("\nRetrieving Explanations:")
    for finding in positive_findings:
        # Retrieve similar NLEs from training data
        similar_nles = retrieve_similar_sentences(finding, dataset)
        
        for nle in similar_nles[:3]:
            print(f"  For {finding}:")
            print(f"    - '{nle}'")
    
    # Alternative: Generate explanation with a seq2seq model
    # predicted_nle = text_generator(h_image, finding)
    # print(f"Generated: {predicted_nle}")
```

---

## 8. Key Implementation Checklist

```
□ Load MIMIC-CXR directory structure (p10-p19)
□ Extract sentences using spaCy sentencizer
□ Create sentence_ID format: "{report_ID}#{index}"
□ Load label mappings from encodings.py
  □ MIMIC_DIAGNOSIS2LABEL for diagnosis/evidence
  □ MIMIC_CAT2ONEHOT for image labels
□ Build data loader with 3 label types:
  □ diagnosis_label (one-hot, what this sentence explains)
  □ evidence_label (one-hot, what this sentence supports elsewhere)
  □ img_labels (10×3, visual ground truth for all findings)
□ Implement multi-task loss with weights
□ Train with modality fusion (image + text features)
□ Evaluate explanation quality and image prediction accuracy
□ Generate or retrieve explanations at inference
□ Validate that hedged text correlates with [0,1,0] img_labels
```

