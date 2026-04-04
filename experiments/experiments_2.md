Adding RadLex retrieved knowledge to the RadGraph triplets

### No-hop

NLG metrics

```shell
Results:
  bleu_1: 32.57
  bleu_2: 20.11
  bleu_4: 9.31
  meteor: 30.55
  rouge_l: 30.62
  cider: 60.34
  chain_coverage: 0.0
  avg_hop_depth: 0.0
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

### One-hop RadLex

```shell
Results:
  bleu_1: 32.4
  bleu_2: 19.9
  bleu_4: 9.02
  meteor: 30.44
  rouge_l: 30.29
  cider: 58.38
  chain_coverage: 0.0
  avg_hop_depth: 0.0
  num_samples: 709
```

LLM-as-judge

```shell
Results (99/100 scored):
  clinical_accuracy: 2.79/5
  completeness: 2.09/5
  reasoning_quality: 2.96/5
  language_quality: 4.1/5
  overall: 2.98/5
```
