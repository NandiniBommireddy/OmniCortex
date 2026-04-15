# Experiment

- Adding RadLex/PrimeKG reasoning chains to the RadGraph triplets.
- LLM as judge on 100/709 eval samples.

# Metrics Reference

### NLG Metrics (lexical overlap with reference text)

- **BLEU-1/2/4**: n-gram precision between generated and reference text. Higher = more word overlap.
- **METEOR**: Unigram recall + precision with synonym matching. More forgiving than BLEU.
- **ROUGE-L**: Longest common subsequence F1. Measures sentence-level structure similarity.
- **CIDEr**: TF-IDF weighted n-gram similarity. Rewards terms distinctive to the reference corpus.

### Clinical Entity Metrics

- **Entity Recall**: % of reference clinical entities (from a fixed 28-term vocabulary) mentioned in the generated text.
- **Hallucination Rate**: % of generated clinical entities NOT present in the reference. Lower = better.
- **RadGraph Precision**: Of clinical entities extracted by RadGraph from generated text, how many also appear in the reference. Higher = less hallucination.
- **RadGraph Recall**: Of clinical entities extracted by RadGraph from reference text, how many appear in the generated text. Higher = more complete.
- **RadGraph F1**: Harmonic mean of RadGraph Precision and Recall. Overall clinical entity fidelity.

### LLM-as-Judge (Claude Haiku, 1-5 scale)

- **Clinical Accuracy**: Are the clinical findings correct and supported by the image?
- **Completeness**: Does the explanation cover all relevant findings from the reference?
- **Reasoning Quality**: Are causal links and clinical reasoning sound?
- **Language Quality**: Is the text fluent, clear, and using appropriate medical terminology?
- **Overall**: Holistic assessment of explanation quality.

---

## liuhaotian/llava-v1.5-7b

### Base (no-hop)

NLG

```shell
Results:
  bleu_1: 32.07
  bleu_2: 19.77
  bleu_4: 8.82
  meteor: 30.57
  rouge_l: 30.76
  cider: 58.41
  entity_recall: 60.15
  hallucination_rate: 39.53
  radgraph_precision: 42.91
  radgraph_recall: 42.44
  radgraph_f1: 42.52
  num_samples: 709
```

LLM-as-judge

```shell
Results (99/100 scored):
  clinical_accuracy: 2.75/5
  completeness: 2.03/5
  reasoning_quality: 2.98/5
  language_quality: 4.01/5
  overall: 2.94/5
```

### RadLex

NLG

```shell
Results:
  bleu_1: 34.48
  bleu_2: 22.13
  bleu_4: 10.73
  meteor: 32.85
  rouge_l: 32.67
  cider: 71.88
  entity_recall: 62.21
  hallucination_rate: 35.94
  radgraph_precision: 48.27
  radgraph_recall: 47.72
  radgraph_f1: 47.84
  num_samples: 709
```

LLM-as-judge

```shell
Results (99/100 scored):
  clinical_accuracy: 2.78/5
  completeness: 2.13/5
  reasoning_quality: 2.92/5
  language_quality: 3.99/5
  overall: 2.96/5
```

### PrimeKG

NLG

```shell
Results:
  bleu_1: 35.74
  bleu_2: 23.52
  bleu_4: 11.99
  meteor: 34.26
  rouge_l: 34.45
  cider: 85.05
  entity_recall: 66.61
  hallucination_rate: 36.34
  radgraph_precision: 50.88
  radgraph_recall: 50.75
  radgraph_f1: 50.63
  num_samples: 709
```

LLM-as-judge

```shell
Results (99/100 scored):
  clinical_accuracy: 2.73/5
  completeness: 2.18/5
  reasoning_quality: 2.95/5
  language_quality: 4.1/5
  overall: 2.99/5
```

### Table

#### NLG Metrics

| Metric             | Base (no-hop) | RadLex | PrimeKG |
| ------------------ | ------------- | ------ | ------- |
| BLEU-1             | 32.07         | 35.18  | 35.74   |
| BLEU-2             | 19.77         | 22.87  | 23.52   |
| BLEU-4             | 8.82          | 10.73  | 11.99   |
| METEOR             | 30.57         | 32.85  | 34.26   |
| ROUGE-L            | 30.76         | 32.67  | 34.45   |
| CIDEr              | 58.41         | 71.88  | 85.05   |
| Entity Recall      | 60.15         | 62.21  | 66.61   |
| Hallucination Rate | 39.53         | 35.94  | 36.34   |
| RadGraph Precision | 42.91         | 48.27  | 50.88   |
| RadGraph Recall    | 42.44         | 47.72  | 50.75   |
| RadGraph F1        | 42.52         | 47.84  | 50.63   |
| Num Samples        | 709           | 709    | 709     |

#### LLM-as-Judge (scored 99/100)

| Metric            | Base (no-hop) | RadLex | PrimeKG |
| ----------------- | ------------- | ------ | ------- |
| Clinical Accuracy | 2.75/5        | 2.78/5 | 2.73/5  |
| Completeness      | 2.03/5        | 2.13/5 | 2.18/5  |
| Reasoning Quality | 2.98/5        | 2.92/5 | 2.95/5  |
| Language Quality  | 4.01/5        | 3.99/5 | 4.10/5  |
| Overall           | 2.94/5        | 2.96/5 | 2.99/5  |

## liuhaotian/llava-v1.6-vicuna-7b

### Base

NLG

```shell
Results:
  bleu_1: 32.54
  bleu_2: 20.08
  bleu_4: 9.16
  meteor: 30.5
  rouge_l: 30.55
  cider: 59.6
  entity_recall: 58.51
  hallucination_rate: 39.57
  radgraph_precision: 44.01
  radgraph_recall: 43.77
  radgraph_f1: 43.73
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.73/5
  completeness: 2.13/5
  reasoning_quality: 3.0/5
  language_quality: 4.07/5
  overall: 2.98/5
```

### Radlex

NLG

```shell
Results:
  bleu_1: 33.84
  bleu_2: 21.87
  bleu_4: 10.95
  meteor: 32.46
  rouge_l: 32.45
  cider: 74.28
  entity_recall: 62.1
  hallucination_rate: 36.9
  radgraph_precision: 49.31
  radgraph_recall: 48.74
  radgraph_f1: 48.86
  num_samples: 709
```

LLM-as-judge

```shell
Results (99/100 scored):
  clinical_accuracy: 2.72/5
  completeness: 2.12/5
  reasoning_quality: 2.9/5
  language_quality: 4.02/5
  overall: 2.94/5
```

### PrimeKG

NLG

```shell
Results:
  bleu_1: 36.36
  bleu_2: 24.14
  bleu_4: 12.44
  meteor: 34.92
  rouge_l: 35.01
  cider: 87.5
  entity_recall: 66.91
  hallucination_rate: 34.55
  radgraph_precision: 52.12
  radgraph_recall: 51.74
  radgraph_f1: 51.74
  num_samples: 709
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.76/5
  completeness: 2.17/5
  reasoning_quality: 2.93/5
  language_quality: 4.09/5
  overall: 2.99/5
```

### Table

#### NLG Metrics

| Metric             | Base  | RadLex | PrimeKG |
| ------------------ | ----- | ------ | ------- |
| BLEU-1             | 32.54 | 34.36  | 36.36   |
| BLEU-2             | 20.08 | 22.47  | 24.14   |
| BLEU-4             | 9.16  | 10.95  | 12.44   |
| METEOR             | 30.50 | 32.46  | 34.92   |
| ROUGE-L            | 30.55 | 32.45  | 35.01   |
| CIDEr              | 59.60 | 74.28  | 87.50   |
| Entity Recall      | 58.51 | 62.10  | 66.91   |
| Hallucination Rate | 39.57 | 36.90  | 34.55   |
| RadGraph Precision | 44.01 | 49.31  | 52.12   |
| RadGraph Recall    | 43.77 | 48.74  | 51.74   |
| RadGraph F1        | 43.73 | 48.86  | 51.74   |
| Num Samples        | 709   | 709    | 709     |

#### LLM-as-Judge

| Metric            | Base (100/100) | RadLex (99/100) | PrimeKG (100/100) |
| ----------------- | -------------- | --------------- | ----------------- |
| Clinical Accuracy | 2.73/5         | 2.72/5          | 2.76/5            |
| Completeness      | 2.13/5         | 2.12/5          | 2.17/5            |
| Reasoning Quality | 3.00/5         | 2.90/5          | 2.93/5            |
| Language Quality  | 4.07/5         | 4.02/5          | 4.09/5            |
| Overall           | 2.98/5         | 2.94/5          | 2.99/5            |

## liuhaotian/llava-v1.6-vicuna-13b

### Base (no-hop)

NLG

```shell
Results:
  bleu_1: 32.57
  bleu_2: 20.16
  bleu_4: 9.27
  meteor: 30.75
  rouge_l: 31.31
  cider: 62.51
  entity_recall: 61.65
  hallucination_rate: 39.56
  radgraph_precision: 42.8
  radgraph_recall: 42.31
  radgraph_f1: 42.41
  num_samples: 709
```

LLM-as-judge

```shell
Results (99/100 scored):
  clinical_accuracy: 2.77/5
  completeness: 2.07/5
  reasoning_quality: 2.98/5
  language_quality: 4.1/5
  overall: 2.98/5
```

### Radlex

NLG

```shell
Results:
  bleu_1: 31.99
  bleu_2: 19.37
  bleu_4: 8.55
  meteor: 29.64
  rouge_l: 29.7
  cider: 55.88
  entity_recall: 59.03
  hallucination_rate: 43.01
  radgraph_precision: 42.27
  radgraph_recall: 41.85
  radgraph_f1: 41.94
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.56/5
  completeness: 1.92/5
  reasoning_quality: 2.8/5
  language_quality: 4.01/5
  overall: 2.82/5
```

### PrimeKG

NLG

```shell
Results:
  bleu_1: 37.35
  bleu_2: 24.78
  bleu_4: 12.63
  meteor: 35.98
  rouge_l: 35.82
  cider: 89.05
  entity_recall: 67.79
  hallucination_rate: 33.85
  radgraph_precision: 50.38
  radgraph_recall: 50.21
  radgraph_f1: 50.12
  num_samples: 709
```

LLM-as-judge

```shell
Results (99/100 scored):
  clinical_accuracy: 2.78/5
  completeness: 2.18/5
  reasoning_quality: 2.92/5
  language_quality: 4.06/5
  overall: 2.98/5
```

### Table

#### NLG Metrics

| Metric             | Base (no-hop) | RadLex | PrimeKG |
| ------------------ | ------------- | ------ | ------- |
| BLEU-1             | 32.57         | 32.37  | 37.35   |
| BLEU-2             | 20.16         | 19.79  | 24.78   |
| BLEU-4             | 9.27          | 8.55   | 12.63   |
| METEOR             | 30.75         | 29.64  | 35.98   |
| ROUGE-L            | 31.31         | 30.06  | 35.82   |
| CIDEr              | 62.51         | 55.88  | 89.05   |
| Entity Recall      | 61.65         | 59.03  | 67.79   |
| Hallucination Rate | 39.56         | 43.01  | 33.85   |
| RadGraph Precision | 42.80         | 42.27  | 50.38   |
| RadGraph Recall    | 42.31         | 41.85  | 50.21   |
| RadGraph F1        | 42.41         | 41.94  | 50.12   |
| Num Samples        | 709           | 709    | 709     |

#### LLM-as-Judge

| Metric            | Base (99/100) | RadLex (100/100) | PrimeKG (99/100) |
| ----------------- | ------------- | ---------------- | ---------------- |
| Clinical Accuracy | 2.77/5        | 2.56/5           | 2.78/5           |
| Completeness      | 2.07/5        | 1.92/5           | 2.18/5           |
| Reasoning Quality | 2.98/5        | 2.80/5           | 2.92/5           |
| Language Quality  | 4.10/5        | 4.01/5           | 4.06/5           |
| Overall           | 2.98/5        | 2.82/5           | 2.98/5           |

## microsoft/llava-med-v1.5-mistral-7b

### Base

NLG

```shell
Results:
  bleu_1: 21.71
  bleu_2: 11.38
  bleu_4: 4.07
  meteor: 21.19
  rouge_l: 23.57
  cider: 26.57
  entity_recall: 32.62
  hallucination_rate: 61.95
  radgraph_precision: 42.31
  radgraph_recall: 42.08
  radgraph_f1: 42.17
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.69/5
  completeness: 1.72/5
  reasoning_quality: 2.66/5
  language_quality: 3.6/5
  overall: 2.67/5
```

### RadLex

NLG

```shell
Results:
  bleu_1: 23.62
  bleu_2: 12.57
  bleu_4: 4.09
  meteor: 22.3
  rouge_l: 22.88
  cider: 28.54
  entity_recall: 48.66
  hallucination_rate: 53.31
  radgraph_precision: 29.27
  radgraph_recall: 26.53
  radgraph_f1: 27.56
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.69/5
  completeness: 1.72/5
  reasoning_quality: 2.66/5
  language_quality: 3.6/5
  overall: 2.67/5
```

### PrimeKG

NLG

```shell
Results:
  bleu_1: 25.46
  bleu_2: 14.01
  bleu_4: 4.57
  meteor: 23.53
  rouge_l: 24.01
  cider: 28.01
  entity_recall: 52.01
  hallucination_rate: 49.48
  radgraph_precision: 40.08
  radgraph_recall: 37.81
  radgraph_f1: 38.67
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.74/5
  completeness: 1.82/5
  reasoning_quality: 2.66/5
  language_quality: 3.92/5
  overall: 2.79/5
```

## google/paligemma2-10b-pt-224

### Base

NLG

```shell
Results:
  bleu_1: 31.06
  bleu_2: 17.97
  bleu_4: 7.35
  meteor: 27.25
  rouge_l: 26.88
  cider: 43.95
  entity_recall: 55.16
  hallucination_rate: 47.19
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.61/5
  completeness: 1.82/5
  reasoning_quality: 2.62/5
  language_quality: 3.74/5
  overall: 2.7/5
```

### RadLex

NLG

```shell
Results:
  bleu_1: 32.53
  bleu_2: 20.07
  bleu_4: 8.95
  meteor: 30.42
  rouge_l: 30.23
  cider: 59.14
  entity_recall: 62.04
  hallucination_rate: 41.38
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.71/5
  completeness: 1.98/5
  reasoning_quality: 2.72/5
  language_quality: 4.01/5
  overall: 2.85/5
```

### PrimeKG

```shell
Results:
  bleu_1: 31.31
  bleu_2: 19.02
  bleu_4: 8.44
  meteor: 28.61
  rouge_l: 28.72
  cider: 56.68
  entity_recall: 55.89
  hallucination_rate: 47.15
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.59/5
  completeness: 1.88/5
  reasoning_quality: 2.62/5
  language_quality: 3.81/5
  overall: 2.73/5
```

## microsoft/Phi-3.5-vision-instruct

### Base

NLG

```shell
Results:
  bleu_1: 31.5
  bleu_2: 19.21
  bleu_4: 8.54
  meteor: 29.93
  rouge_l: 30.28
  cider: 58.66
  entity_recall: 60.78
  hallucination_rate: 40.6
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.78/5
  completeness: 2.0/5
  reasoning_quality: 2.85/5
  language_quality: 4.1/5
  overall: 2.93/5
```

### Radlex

NLG

```shell
Results:
  bleu_1: 33.64
  bleu_2: 21.69
  bleu_4: 10.69
  meteor: 32.62
  rouge_l: 32.96
  cider: 75.2
  entity_recall: 61.72
  hallucination_rate: 36.76
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.82/5
  completeness: 2.11/5
  reasoning_quality: 2.89/5
  language_quality: 4.11/5
  overall: 2.98/5
```

### PrimeKG

NLG

```shell
Results:
  bleu_1: 35.01
  bleu_2: 22.37
  bleu_4: 10.69
  meteor: 33.37
  rouge_l: 33.8
  cider: 78.31
  entity_recall: 65.58
  hallucination_rate: 36.31
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.91/5
  completeness: 2.16/5
  reasoning_quality: 2.93/5
  language_quality: 4.13/5
  overall: 3.03/5
```

## meta/Llama-3.2-11B-Vision-Instruct

### Base

NLG

```shell
Results:
  bleu_1: 9.79
  bleu_2: 3.35
  bleu_4: 0.25
  meteor: 10.18
  rouge_l: 10.12
  cider: 4.67
  entity_recall: 17.04
  hallucination_rate: 68.6
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 1.88/5
  completeness: 1.17/5
  reasoning_quality: 1.41/5
  language_quality: 1.19/5
  overall: 1.41/5
```

### Radlex

NLG

```shell
Results:
  bleu_1: 28.68
  bleu_2: 16.27
  bleu_4: 5.98
  meteor: 26.16
  rouge_l: 26.58
  cider: 36.76
  entity_recall: 53.46
  hallucination_rate: 49.28
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.65/5
  completeness: 1.88/5
  reasoning_quality: 2.69/5
  language_quality: 3.72/5
  overall: 2.73/5
```

### PrimeKG

NLG

```shell
Results:
  bleu_1: 34.73
  bleu_2: 23.04
  bleu_4: 12.08
  meteor: 34.49
  rouge_l: 35.2
  cider: 89.36
  entity_recall: 67.62
  hallucination_rate: 34.64
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.98/5
  completeness: 2.22/5
  reasoning_quality: 3.02/5
  language_quality: 4.21/5
  overall: 3.11/5
```

## Qwen/Qwen2.5-VL-7B-Instruct

### Base

NLG

```shell
Results:
  bleu_1: 31.97
  bleu_2: 19.76
  bleu_4: 9.39
  meteor: 30.17
  rouge_l: 30.13
  cider: 64.51
  entity_recall: 56.82
  hallucination_rate: 43.76
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.91/5
  completeness: 2.06/5
  reasoning_quality: 2.94/5
  language_quality: 4.09/5
  overall: 3.0/5
```

### RadLex

NLG

```shell
Results:
  bleu_1: 25.68
  bleu_2: 14.15
  bleu_4: 5.36
  meteor: 24.23
  rouge_l: 24.26
  cider: 36.29
  entity_recall: 45.51
  hallucination_rate: 54.53
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.57/5
  completeness: 1.8/5
  reasoning_quality: 2.6/5
  language_quality: 3.85/5
  overall: 2.71/5
```

### PrimeKG

NLG

```shell
Results:
  bleu_1: 35.36
  bleu_2: 23.15
  bleu_4: 11.61
  meteor: 34.47
  rouge_l: 34.59
  cider: 86.92
  entity_recall: 66.71
  hallucination_rate: 34.85
  num_samples: 709
```

LLM-as-judge

```shell
Results (100/100 scored):
  clinical_accuracy: 2.98/5
  completeness: 2.22/5
  reasoning_quality: 3.07/5
  language_quality: 4.15/5
  overall: 3.1/5
```

---

# Table

## NLG Metrics (All Models)

| Model                    | Condition | BLEU-4 | METEOR | ROUGE-L | CIDEr | Entity Recall | Hallucination Rate |
| ------------------------ | --------- | ------ | ------ | ------- | ----- | ------------- | ------------------ |
| **LLaVA-1.5-7B**         | Base      | 8.82   | 30.57  | 30.76   | 58.41 | 60.15         | 39.53              |
|                          | RadLex    | 10.73  | 32.85  | 32.67   | 71.88 | 62.21         | **35.94**          |
|                          | PrimeKG   | **11.99** | **34.26** | **34.45** | **85.05** | **66.61** | 36.34              |
| **LLaVA-1.6-Vicuna-7B**  | Base      | 9.16   | 30.50  | 30.55   | 59.60 | 58.51         | 39.57              |
|                          | RadLex    | 10.95  | 32.46  | 32.45   | 74.28 | 62.10         | 36.90              |
|                          | PrimeKG   | **12.44** | **34.92** | **35.01** | **87.50** | **66.91** | **34.55**          |
| **LLaVA-1.6-Vicuna-13B** | Base      | 9.27   | 30.75  | 31.31   | 62.51 | 61.65         | 39.56              |
|                          | RadLex    | 8.55   | 29.64  | 29.7    | 55.88 | 59.03         | 43.01              |
|                          | PrimeKG   | **12.63** | **35.98** | **35.82** | **89.05** | **67.79** | **33.85**          |
| **LLaVA-Med-7B**         | Base      | 4.07   | 21.19  | 23.57   | 26.57 | 32.62         | 61.95              |
|                          | RadLex    | 4.09   | 22.3   | 22.88   | **28.54** | 48.66      | 53.31              |
|                          | PrimeKG   | **4.57** | **23.53** | **24.01** | 28.01 | **52.01**     | **49.48**          |
| **PaLI-Gemma2-10B**      | Base      | 7.35   | 27.25  | 26.88   | 43.95 | 55.16         | 47.19              |
|                          | RadLex    | **8.95** | **30.42** | **30.23** | **59.14** | **62.04** | **41.38**          |
|                          | PrimeKG   | 8.44   | 28.61  | 28.72   | 56.68 | 55.89         | 47.15              |
| **Phi-3.5-Vision**       | Base      | 8.54   | 29.93  | 30.28   | 58.66 | 60.78         | 40.60              |
|                          | RadLex    | **10.69** | 32.62  | 32.96   | 75.2  | 61.72         | 36.76              |
|                          | PrimeKG   | **10.69** | **33.37** | **33.80** | **78.31** | **65.58** | **36.31**          |
| **Llama-3.2-11B-Vision** | Base      | 0.25   | 10.18  | 10.12   | 4.67  | 17.04         | 68.60              |
|                          | RadLex    | 5.98   | 26.16  | 26.58   | 36.76 | 53.46         | 49.28              |
|                          | PrimeKG   | **12.08** | **34.49** | **35.20** | **89.36** | **67.62** | **34.64**          |
| **Qwen2.5-VL-7B**        | Base      | 9.39   | 30.17  | 30.13   | 64.51 | 56.82         | 43.76              |
|                          | RadLex    | 5.36   | 24.23  | 24.26   | 36.29 | 45.51         | 54.53              |
|                          | PrimeKG   | **11.61** | **34.47** | **34.59** | **86.92** | **66.71** | **34.85**          |

## LLM-as-Judge Metrics (All Models)

| Model                    | Condition | Clinical Accuracy | Completeness | Reasoning Quality | Language Quality | Overall |
| ------------------------ | --------- | ----------------- | ------------ | ----------------- | ---------------- | ------- |
| **LLaVA-1.5-7B**         | Base      | 2.75              | 2.03         | **2.98**          | 4.01             | 2.94    |
|                          | RadLex    | **2.78**          | 2.13         | 2.92              | 3.99             | 2.96    |
|                          | PrimeKG   | 2.73              | **2.18**     | 2.95              | **4.10**         | **2.99** |
| **LLaVA-1.6-Vicuna-7B**  | Base      | 2.73              | 2.13         | **3.00**          | 4.07             | 2.98    |
|                          | RadLex    | 2.72              | 2.12         | 2.90              | 4.02             | 2.94    |
|                          | PrimeKG   | **2.76**          | **2.17**     | 2.93              | **4.09**         | **2.99** |
| **LLaVA-1.6-Vicuna-13B** | Base      | 2.77              | 2.07         | **2.98**          | **4.10**         | **2.98** |
|                          | RadLex    | 2.56              | 1.92         | 2.80              | 4.01             | 2.82    |
|                          | PrimeKG   | **2.78**          | **2.18**     | 2.92              | 4.06             | **2.98** |
| **LLaVA-Med-7B**         | Base      | 2.69              | 1.72         | **2.66**          | 3.60             | 2.67    |
|                          | RadLex    | 2.69              | 1.72         | **2.66**          | 3.60             | 2.67    |
|                          | PrimeKG   | **2.74**          | **1.82**     | **2.66**          | **3.92**         | **2.79** |
| **PaLI-Gemma2-10B**      | Base      | 2.61              | 1.82         | 2.62              | 3.74             | 2.70    |
|                          | RadLex    | **2.71**          | **1.98**     | **2.72**          | **4.01**         | **2.85** |
|                          | PrimeKG   | 2.59              | 1.88         | 2.62              | 3.81             | 2.73    |
| **Phi-3.5-Vision**       | Base      | 2.78              | 2.00         | 2.85              | 4.10             | 2.93    |
|                          | RadLex    | 2.82              | 2.11         | 2.89              | 4.11             | 2.98    |
|                          | PrimeKG   | **2.91**          | **2.16**     | **2.93**          | **4.13**         | **3.03** |
| **Llama-3.2-11B-Vision** | Base      | 1.88              | 1.17         | 1.41              | 1.19             | 1.41    |
|                          | RadLex    | 2.65              | 1.88         | 2.69              | 3.72             | 2.73    |
|                          | PrimeKG   | **2.98**          | **2.22**     | **3.02**          | **4.21**         | **3.11** |
| **Qwen2.5-VL-7B**        | Base      | 2.91              | 2.06         | 2.94              | 4.09             | 3.00    |
|                          | RadLex    | 2.57              | 1.80         | 2.60              | 3.85             | 2.71    |
|                          | PrimeKG   | **2.98**          | **2.22**     | **3.07**          | **4.15**         | **3.10** |
