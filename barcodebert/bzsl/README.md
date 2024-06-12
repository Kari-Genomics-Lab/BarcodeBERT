# Bayesian Zero-shot Learning Task

We use Bayesian zero-shot learning (BZSL) as a downstream task for evaluation of the BarcodeBERT model.
This method is taken from "Fine-Grained Zero-Shot Learning with DNA as Side Information" and "Classifying the unknown: Insect identification with deep hierarchical Bayesian learning". Our main modifications include adapting the BZSL
approach to run natively in Python and to work with other models for DNA feature extraction, including BarcodeBERT,
DNABERT, and DNABERT-2, to compare against the baseline CNN used in the original paper. The original repositories for
this method can be found [here](https://github.com/sbadirli/Fine-Grained-ZSL-with-DNA) and [here](https://github.com/sbadirli/Zero-shot-Insect-Discovery).

## Datasets

The datasets used for our BZSL evaluation came from the Badirli papers. As of writing, you can find these datasets
at the below links:

INSECT dataset
* [Features and splits](https://www.dropbox.com/sh/gt6tkech0nvftk5/AADOUJc_Bty3sqOsqWHxhmULa?dl=0)
* [Images](https://indiana-my.sharepoint.com/:f:/g/personal/sbadirli_iu_edu/Ek2KDBxTndlFl_7XblTL-8QBZ6b0C0izgDJIBJQlWtiRKA?e=bCfCMH)

Badirli 2023 dataset
* [Features and splits](https://dataworks.iupui.edu/handle/11243/41)

## Model Weights

Model weights:
* BarcodeBERT - please see the main [README](../README.md) for more information on obtaining the pretrained weights
* [DNABERT](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing) (k=6), as from [here](https://github.com/jerryji1993/DNABERT)
* DNABERT-2 - extracted directly from huggingface repository

## Usage (surrogate-species prediction)

The below instructions detail how to reproduce the results in the paper for the surrogate species BZSL task (Badirli et. al. 2021), i.e. predicting a surrogate species class for unseen species at test time.

Please use the [run.sh](./run.sh) script to run all of the parts at once. You can run
```shell
./run.sh -h
```
to find more information on the supported arguments.

If you choose to run manually, you will need to generate the DNA features from the pretrained model or the finetuned model, and then subsequently pass those features to the Bayesian model.

### Fine-tuning

If you want to finetune either BarcodeBERT or DNABERT, first run the following:
```shell
DATA=path/to/data/dir
MODEL=model_name  # one of ["barcodebert", "dnabert", "dnabert2"]
OUTPUT=path/to/output/dir
CHECKPOINT=path/to/model_ckpt
KMER=kmer_size_for_tokenization

python finetuning/supervised_learning.py --input_path "$DATA/res101.mat" --model "$MODEL" --output_dir "$OUTPUT/finetuning/$MODEL" --n_epoch 12 --checkpoint "$CHECKPOINT" -k $KMER --model-output "$OUTPUT/finetuning/$MODEL/supervised_model.pth"
```

This will generate a CSV of DNA feature embeddings which you can then use in the BZSL task. Alternatively, you can use
the `feature_extraction/main.py` script itself to extract the features for the base model after finetuning.

For any finetuning related to the DNABERT-2 model, you will need to use the [DNABERT-2](https://github.com/Zhihan1996/DNABERT_2) repository:
```shell
DATA=path/to/data/dir
MODEL=dnabert2
OUTPUT=path/to/output/dir

cd models/dnabert2/finetune
python create_dataset.py --input_path "$DATA/res101.mat" --output "$DATA"
python train.py \
  --model_name_or_path zhihan1996/DNABERT-2-117M \
  --data_path "$DATA" \
  --kmer -1 \
  --run_name DNABERT2_FINETUNING \
  --model_max_length 265 \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 20 \
  --fp16 \
  --save_steps 200 \
  --output_dir "$OUTPUT/finetuning/$MODEL/" \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --warmup_steps 50 \
  --logging_steps 100 \
  --overwrite_output_dir True \
  --log_level info \
  --find_unused_parameters False \
  --eval_and_save_results False \
  --save_model True
```

### Feature Extraction

To extract the DNA features for any of the three models, run
```shell
DATA=path/to/data/dir
MODEL=model_name  # one of ["barcodebert", "dnabert", "dnabert2"]
OUTPUT=path/to/output/dir
CHECKPOINT=path/to/model_ckpt
KMER=kmer_size_for_tokenization
EMBEDDINGS="$OUTPUT/embeddings/dna_embeddings_$MODEL.csv"

python feature_extraction/main.py --input_path "$DATA/res101.mat" --model "$MODEL" --checkpoint "$CHECKPOINT" --output "$EMBEDDINGS" -k "$KMER"
```

If you finetune the model first, the finetuning code will generate this for you already for BarcodeBERT and DNABERT only.

For DNABERT-2, prior to finetuning, you do not need to specify a checkpoint, as the  code will by default use the model weights from Huggingface.

### Bayesian Zero-shot Learning

To perform a hyperparameter search over the Bayesian model parameters and generate the final metrics, run
```shell
DATA=path/to/data/dir
MODEL=model_name  # one of ["barcodebert", "dnabert", "dnabert2"]
OUTPUT=path/to/output/dir
EMBEDDINGS="$OUTPUT/embeddings/dna_embeddings_$MODEL.csv"
HP_OUTPUT="$OUTPUT/results/bzsl_output_$MODEL.json"

python surrogate_species/main.py --datapath "$DATA" --embeddings "$EMBEDDINGS" --tuning --output "$HP_OUTPUT"
```

## Usage (genus-species prediction)

To run our models against the genus-species BZSL task (Badirli et. al. 2023), i.e. predicting the genus of a sample for unseen (or undescribed) species, first generate the DNA features as follows:

```shell
DATA=path/to/data/dir
MODEL=model_name  # one of ["barcodebert", "dnabert", "dnabert2"]
OUTPUT=path/to/output/dir
CHECKPOINT=path/to/model_ckpt
KMER=kmer_size_for_tokenization
EMBEDDINGS="$OUTPUT/embeddings/dna_embeddings_$MODEL.csv"

python feature_extraction/main.py --input_path "$DATA/res101.mat" --model "$MODEL" --checkpoint "$CHECKPOINT" --output "$EMBEDDINGS" -k "$KMER" --save-all
```

Note that the `--save-all` argument is required to generate DNA features for every sample rather than aggregate by class.

Afterward, run the BZSL code as follows:

```shell
DATA=path/to/data/dir
MODEL=bzsl_model_type  # one of ["OSBC_DNA", "OSBC_IMG", "OSBC_DIC", "OSBC_DIL", "OSBC_DIT"]
OUTPUT=path/to/output/dir
EMBEDDINGS="$OUTPUT/embeddings/dna_embeddings_$MODEL.csv"

python genus_species/main.py --datapath "$DATA" --model --embeddings "$EMBEDDINGS" --tuning
```
