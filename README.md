# Tumor Classification
This repo is for UIUC CS 466 Introduction to Bioinformatics course project. Multi-classification of Tumor.

## Dataset
A dataset from UCI Macine Learning Repository, consists of a random extraction of gene expressions of patients having FIVE types of tumor, BRCA, KIRC, COAD, LUAD, PRAD.

Data Source: [Tumor dataset](https://archive.ics.uci.edu/ml/datasets/primary+tumor)


## How to run:
```
$python main.py --method --pca_flag
```

`--method`: `sup` supervised: random forest. `unsup` unsupervised clustering: kmeans


`--pca_flag': `y` apply pca. `n` don't apply pca
