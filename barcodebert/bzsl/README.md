# Bayesian Zero-shot Learning Task

We use Bayesian zero-shot learning (BZSL) as a downstream task for evaluation of the BarcodeBERT model.
This method is taken from "Fine-Grained Zero-Shot Learning with DNA as Side Information" and "Classifying the unknown: Insect identification with deep hierarchical Bayesian learning". Our main modifications include adapting the BZSL 
approach to run natively in Python and to work with other models for DNA feature extraction, including BarcodeBERT, 
DNABERT, and DNABERT-2, to compare against the baseline CNN used in the original paper. The original repositories for
this method can be found [here](https://github.com/sbadirli/Fine-Grained-ZSL-with-DNA) and [here](https://github.com/sbadirli/Zero-shot-Insect-Discovery).

## Setup

This repository was tested with Python 3.11.

To setup your environment, create a virtual environment and run the following:
```
pip install -r requirements.txt
pip uninstall triton  # required because triton has some backwards incompatibility issues with this repo
```

### Datasets

The datasets used for our BZSL evaluation came from the Badirli papers. As of writing, you can find these datasets
at the below links:

INSECT dataset
* [Features and splits](https://www.dropbox.com/sh/gt6tkech0nvftk5/AADOUJc_Bty3sqOsqWHxhmULa?dl=0)
* [Images](https://indiana-my.sharepoint.com/:f:/g/personal/sbadirli_iu_edu/Ek2KDBxTndlFl_7XblTL-8QBZ6b0C0izgDJIBJQlWtiRKA?e=bCfCMH)

Badirli 2023 dataset
* [Features and splits](https://dataworks.iupui.edu/handle/11243/41)

### Model Weights

Model weights:
* BarcodeBERT - please see the main [README](../README.md) for more information on obtaining the pretrained weights
* [DNABERT](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing) (k=6), as from [here](https://github.com/jerryji1993/DNABERT)
* DNABERT-2 - extracted directly from huggingface repository

### Usage (surrogate-species prediction)

The below instructions detail how to reproduce the results in the paper for the surrogate species BZSL task (Badirli et. al. 2021), i.e. predicting a surrogate species class for unseen species at test time.

Please use the [run.sh](./run.sh) script to run all of the parts at once. If you choose to run manually, you can follow
the steps in the shell script or follow the steps below.

For any finetuning related to the DNABERT-2 model, you will need to use the [DNABERT-2](https://github.com/Zhihan1996/DNABERT_2) repository. A submodule for this repository has been created here for your convenience.

If you want to finetune the model, first run the following:
```
python finetuning/train.py 
```

This will generate a CSV of DNA feature embeddings which you can then use in the BZSL task. Alternatively, you can use
the `feature_extraction/main.py` script itself to extract the features for the base model. 

To perform a hyperparameter search over the Bayesian model parameters and generate the final metrics, run
```
python surrogate_species/main.py <options>
```

### Usage (genus-species prediction)

The below instructions detail how to run our models against the genus-species BZSL task (Badirli et. al. 2023), i.e. predicting the genus of a sample for unseen (or undescribed) species.