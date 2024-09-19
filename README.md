# CLERA - Cellular Latent Equations Representation and Analysis

CLERA is a novel end-to-end computational framework designed to uncover parsimonious dynamical models and identify active gene programs from single-cell RNA sequencing data. This repository contains the code used to train and demonstrate CLERA on three scRNA datasets.

## Usage
### Training
Navigate to the appropriate example dataset directory (e.g., Pancreas or Bone_Marrow) under Examples to start the training. Each directory contains the notebook which loads the dataset, preprocesses it and start training. Modify the params dictionary for more control over the training trajectory

### Inference
After training, navigate to the Inference directory to:
1. Select the best experiment
2. Find SHAP values for the best experiment
3. Create interaction networks based on the learned latent variables
Make sure to choose the appropriate path variables for inference based on the dataset used for training.

## Installation
This code is supported with Python 3.6.7 Run the following command to install the required dependencies:
pip install -r requirements.txt


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
|   ├── Inference
        ├── Choose Best Experiment
        ├── Find SHAP values
        └── Create Interaction Network
