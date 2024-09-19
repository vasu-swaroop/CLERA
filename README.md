# CLERA - Cellular Latent Equations Representation and Analysis

CLERA is a novel end-to-end computational framework designed to uncover parsimonious dynamical models and identify active gene programs from single-cell RNA sequencing data. This repository contains the code used to train and demonstrate CLERA on three scRNA datasets.

## Structure

```bash
.
├── src
│   ├── Training scripts
│   ├── Utility Files
├── Examples
│   ├── Pancreas
│   ├── Bone_Marrow
│   ├── SERGIO
└── Inference
    ├── Choose Best Experiment
    ├── Find SHAP values
    └── Create Interaction Network
