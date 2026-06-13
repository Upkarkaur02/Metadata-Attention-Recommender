# MetaAttnRec: Metadata-Attention Recommender

Official implementation of the paper:

**Metadata-Attentive Neural Recommendation with Feature-Level Attribution for Restricted Cold-Start Movie Recommendation**

---

## Overview

Recommender systems often struggle with cold-start items that lack historical interaction data. This project introduces a metadata-attentive neural recommendation framework that leverages movie metadata and feature-level attention to improve recommendation quality under a restricted cold-start setting.

The proposed approach combines:

- Neural Collaborative Filtering
- Movie Genre Metadata
- Tag Genome Semantic Descriptors
- Feature-Level Metadata Attention
- Cold-Start Recommendation Evaluation
- Attention-Based Item Attribution Analysis

Experiments are conducted on the MovieLens 20M dataset with Tag Genome metadata.

---

---

## Dataset

This work uses:

### MovieLens 20M

https://grouplens.org/datasets/movielens/20m/

### Tag Genome Metadata

https://grouplens.org/datasets/movielens/tag-genome/

The dataset contains:

- 20 million ratings
- Movie genre information
- Tag Genome semantic relevance scores

---

## Methodology

### Baseline Neural Recommender

The baseline model learns user and item embeddings and predicts ratings using fully connected layers.

### Metadata-Attention Recommender

The proposed model incorporates:

- User embeddings
- Item embeddings
- Genre metadata
- Tag Genome features
- Feature-level metadata attention

The attention mechanism assigns weights to metadata dimensions and generates an attention-weighted item representation used for prediction.

---

## Experimental Protocol

A restricted cold-start setting is adopted:

- Selected cold-start items are removed from training interactions.
- Metadata remains available for all items.
- Models must rely on metadata when collaborative item history is unavailable.

Evaluation is performed on:

- Warm-start items
- Cold-start items
- Top-N recommendation quality

---

## Results

| Metric | Baseline | Attention Model |
|----------|----------|----------|
| Warm RMSE | 1.0249 | 0.9629 |
| Warm MAE | 0.8072 | 0.7393 |
| Cold RMSE | 1.1861 | 0.9628 |
| Cold MAE | 0.9891 | 0.7471 |
| Precision@10 | 0.6817 | 0.6886 |
| Recall@10 | 0.3045 | 0.3123 |
| Hit Ratio@10 | 0.9945 | 0.9950 |
| NDCG@10 | 0.8571 | 0.8620 |

Key findings:

- 6.05% reduction in Warm RMSE
- 18.83% reduction in Cold RMSE
- 24.47% reduction in Cold MAE
- Consistent improvements across ranking metrics

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Upkarkaur02/MetaAttnRec.git
cd MetaAttnRec
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

### Baseline Model

```bash
python baseline_model.py
```

### Metadata-Attention Model

```bash
python attention_model.py
```

### Cyclic Dual Model

```bash
python cyclic_dual_model.py
```

---

## Evaluation

Generate ranking metrics:

```bash
python evaluate_topk.py
```

Compute percentage improvements:

```bash
python calculate_improvements.py
```

---

## Explainability Analysis

Extract metadata attention weights:

```bash
python decode_attention.py
```

Generate cold-start recommendation case studies:

```bash
python cold_start_case_study.py
```

---

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib

---

## Reproducibility

All experiments use:

- MovieLens 20M
- Tag Genome metadata
- Restricted cold-start protocol
- Early stopping based on validation loss
- Adam optimizer

---

## Paper

The full research paper is available in the 'Research Paper' directory.

---

## Authors

- Upkar Kaur
- Palak Mahajan
- Santushti Verma
- Sukhpal Singh

---
