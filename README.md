# Conversational Question Answering using Dense Passage Retrieval and Fine-tuning with LoRA

## Overview

This project presents an implementation of a Conversational Question Answering (QA) system based on Dense Passage Retrieval (DPR), enhanced through fine-tuning with Low-Rank Adaptation (LoRA). Our approach builds upon the foundational work introduced by Karpukhin et al. in the paper ["Dense Passage Retrieval for Open-Domain Question Answering"](https://arxiv.org/pdf/2004.04906).

Dense Passage Retrieval significantly improves open-domain QA by effectively encoding and retrieving relevant passages from large textual corpora. This project adapts DPR specifically to conversational contexts, where subsequent questions rely on the conversational history. Furthermore, we employ LoRA for efficient parameter-efficient fine-tuning, thereby improving the model's performance with minimal computational overhead.

## Objectives

- Evaluate DPR adapted for Conversational Question Answering scenarios.
- Optimize fine-tuning efficiency through Low-Rank Adaptation (LoRA).
- Evaluate and compare the performance improvements achieved through fine-tuning.

## Methodology

### Dense Passage Retrieval (DPR)
DPR consists of two main components:
- **Passage Encoder**: Encodes textual passages into dense embeddings.
- **Query Encoder**: Converts questions into dense vector representations for efficient retrieval.

This approach allows rapid, scalable retrieval of passages relevant to the input queries.

### Conversational Adaptation
- Incorporates conversational history to better understand the context of follow-up questions.
- Utilizes concatenated conversational context embeddings to refine query representations.

### Fine-Tuning with LoRA
- Low-Rank Adaptation (LoRA) introduces trainable, low-rank matrices to existing pretrained models.
- Efficiently fine-tunes DPR models, significantly reducing computational cost and resource usage.

## Experimental Setup

- Dataset: MSMARCO.
- Evaluation Metrics: Top 5 accuracy, Top 20 Accuracy, Top 100 Accuracy.
- Baseline Comparison: Standard DPR model performance vs. fine-tuned DPR with LoRA.

## Results

Fine-tuning DPR with LoRA demonstrated improvements over the baseline, validating the effectiveness of the approach presented in the original paper.



## References
- Original DPR Paper: [Karpukhin et al., 2020](https://arxiv.org/pdf/2004.04906)
- LoRA Methodology: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)

## Contributors
- Eva Robillard, Lucas Murray


