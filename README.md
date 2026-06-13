# Metadata-Attention-Recommendor

Official implementation of:

**Metadata-Attentive Neural Recommendation with Feature-Level Attribution for Restricted Cold-Start Movie Recommendation**

## Overview

This repository contains the implementation of a metadata-aware neural recommendation framework that combines:

- Neural Collaborative Filtering
- Genre Metadata
- Tag Genome Semantic Features
- Feature-Level Attention
- Restricted Cold-Start Evaluation

Experiments are conducted on the MovieLens 20M dataset.

## Dataset

MovieLens 20M Dataset:

https://grouplens.org/datasets/movielens/20m/

## Requirements

```bash
pip install -r requirements.txt
```

## Training

Baseline model:

```bash
python baseline_model.py
```

Attention model:

```bash
python attention_model.py
```

Cyclic Dual model:

```bash
python cyclic_dual_model.py
```

## Evaluation

```bash
python evaluate_topk.py
```

## Explainability

```bash
python decode_attention.py
```

## Paper

The associated research paper is available in:

```text
paper/RESEARCH_PAPER.pdf
```

## Citation

```bibtex
@article{kaur2026metaattnrec,
  title={Metadata-Attentive Neural Recommendation with Feature-Level Attribution for Restricted Cold-Start Movie Recommendation},
  author={Upkar Kaur, Palak Mahajan, Santushti Verma and Sukhpal Singh},
  year={2026}
}
```
