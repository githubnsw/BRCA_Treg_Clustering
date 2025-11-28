# BRCA Treg-Enriched Subtype Clustering
This repository contains the core codebase used in our study to identify and externally validate a Treg-enriched breast cancer subtype using multi-omics integration, autoencoder-based dimensionality reduction, and consensus K-means clustering.

The repository provides:
- Preprocessed multi-omics matrices (TCGA BRCA discovery cohort)
- Autoencoder training script (PyTorch)
- Latent embedding matrix used for clustering
- Consensus clustering pipeline (R)
- Silhouette/PAC evaluation scripts

---

## 🔥 Overview
We constructed an integrative multi-omics matrix from TCGA BRCA RNA-seq, miRNA, protein, and methylation data. After preprocessing, an autoencoder (AE) model was trained to learn a compressed latent representation. This latent embedding was used for consensus K-means clustering (K = 3), revealing a robust Treg-enriched tumor subtype.

The same pipeline (same AE model & clustering workflow) was applied to the GSE96058 external dataset for independent validation.

---
