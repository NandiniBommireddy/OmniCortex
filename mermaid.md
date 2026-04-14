Base paper workflow
```mermaid
graph LR
      subgraph "KG-LLaVA (Base Paper)"
          A[MIMIC-NLE + CXR Reports] -->|RadGraph| B[RadGraph triplets]
          B -->|MedCLIP + FAISS| C[Top-k retrieved triplets]
          C -->|Triplets injected in prompt| D[LLaVA JSON]
          D -->|LoRA fine-tuning<br/>single model, single condition| E[Fine-tuned LLaVA]
          E -->|Generate explanations| F[Generated answers]
          F -->|BLEU, ROUGE, CIDEr| G[NLG Metrics only]
      end
```

Our approach

```mermaid
graph TB
      subgraph "Phase 1: Data Extraction"
          A[MIMIC-NLE + CXR Reports] -->|RadGraph modern-xl<br/>.venv-radgraph| B[RadGraph triplets<br/>e.g. opacity suggestive_of
  pneumonia]
      end

      subgraph "Phase 2: Retrieval"
          B -->|MedCLIP embeddings<br/>FAISS nearest-neighbor| C[Top-k retrieved triplets<br/>per image]
      end

      subgraph "Phase 3: KG Chain Construction"
          B --> D1[No chains<br/>triplets only]

          B -->|Fuzzy match entities<br/>to RadLex concepts| E1[RadLex entity alignment]
          E1 -->|Traverse May_Cause edges<br/>in Gamuts ontology| F1[RadLex reasoning chains<br/>e.g. pneumonia may cause fever]

          B -->|CUI linking via<br/>scispaCy + Neo4j| E2[PrimeKG entity alignment]
          E2 -->|Query disease-disease and<br/>phenotype-phenotype edges| F2[PrimeKG reasoning chains<br/>e.g. edema is associated with
   dyspnea]
      end

      subgraph "Phase 4: Prompt Assembly"
          C --> G
          D1 -->|Triplets only in prompt| G[Baseline LLaVA JSON]

          C --> H
          F1 -->|Triplets + chains in prompt| H[RadLex LLaVA JSON]

          C --> I
          F2 -->|Triplets + chains in prompt| I[PrimeKG LLaVA JSON]
      end

      subgraph "Phase 5: Fine-tuning"
          G --> J["LoRA fine-tuning<br/>4 models × 3 variants = 12 runs<br/>1 epoch, A100 GPU via Modal"]
          H --> J
          I --> J
          J --> K[12 LoRA checkpoints]
      end

      subgraph "Phase 6: Inference"
          K -->|Generate explanations<br/>for 709 test images| L[12 sets of<br/>generated answers]
      end

      subgraph "Phase 7: Evaluation"
          L -->|BLEU, ROUGE-L, METEOR, CIDEr<br/>lexical overlap with reference| M[NLG Metrics]
          L -->|Run RadGraph on both<br/>generated + reference text<br/>compare entity sets| N[RadGraph F1]
          L -->|Claude Haiku scores<br/>clinical accuracy, reasoning<br/>language, overall on 1-5| O[LLM-as-Judge]
      end

      style D1 fill:#e8e8e8
      style F1 fill:#d4edda
      style F2 fill:#cce5ff
      style M fill:#fff3cd
      style N fill:#fff3cd
      style O fill:#fff3cd
```



Model                       | Backbone  | Scale | Domain
-------------------- |-----------|-------|----------
LLaVA-1.5-7B      | LLaMA     |  7B   | General
LLaVA-1.6-7B      | Vicuna    |  7B   | General
LLaVA-1.6-13B     | Vicuna    | 13B   | General
LLaVA-Med-7B      | Mistral   |  7B   | Biomedical
