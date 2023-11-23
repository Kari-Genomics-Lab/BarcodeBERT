#!/bin/bash
set -e

FINETUNE=false
BATCH_SIZE=32
KMER=6

# parse arguments
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--checkpoint)
      CHECKPOINT="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--data)
      DATA="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output)
      OUTPUT="$2"
      shift # past argument
      shift # past value
      ;;
    -k|--kmer)
      KMER="$2"
      shift # past argument
      shift # past value
      ;;
    --finetune)
      FINETUNE="true"
      shift # past value
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      echo "Usage:"
      echo "  -m, --model        name of model to use for generating DNA embeddings."
      echo "                     Must be one of [barcodebert, dnabert, dnabert2]."
      echo "  -c, --checkpoint   path to model checkpoint. Only required for BarcodeBERT "
      echo "                     and DNABERT."
      echo "  -d, --data         path to folder containing input data. Must contain files "
      echo "                     named res101.mat and att_splits.mat."
      echo "  -o, --output       path to output prefix folder in which to save embedding "
      echo "                     and finetuning results"
      echo "  -k, --kmer         size of k-mer to use in tokenization. Only relevant for "
      echo "                     BarcodeBERT and DNABERT. By default, set to 5 for BarcodeBERT "
      echo "                     and 6 for DNABERT."
      echo "  --finetune         If specified, will finetune model prior to BZSL tuning"
      exit 0
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [ -z "$KMER" ]; then
    if [ $MODEL = "barcodebert" ]; then
        KMER=5
    elif [ $MODEL = "dnabert" ]; then
        KMER=6
    fi
fi

# Paths
ROOT="$(pwd)"
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT="$(realpath $CHECKPOINT)"
fi
DATA="$(realpath $DATA)"
OUTPUT="$(realpath $OUTPUT)"

# Fine tune model
if [ $FINETUNE = true ]; then
    EMBEDDINGS="$OUTPUT/finetuning/$MODEL/dna_embedding_supervised.csv"
    mkdir -p "$OUTPUT/finetuning/$MODEL"

    # Fine-tune DNABERT-2; separate script needs to be run
    if [ $MODEL = "dnabert2" ]; then
        cd $ROOT/dnabert2/finetune
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
        
        # generate embeddings for DNABERT-2
        CHECKPOINT="$OUTPUT/finetuning/$MODEL"
        cd $ROOT/DNA_Embeddings
        mkdir -p "$OUTPUT/embeddings"
        python bert_extract_dna_feature.py --input_path "$DATA/res101.mat" --model "$MODEL" --checkpoint "$CHECKPOINT" --output "$EMBEDDINGS"
    
    # Fine-tune BarcodeBERT or DNABERT
    else
        cd $ROOT/DNA_Embeddings
        if [ -n "$CHECKPOINT" ]; then
            python supervised_learning.py --input_path "$DATA/res101.mat" --model "$MODEL" --output_dir "$OUTPUT/finetuning/$MODEL" --n_epoch 12 --checkpoint "$CHECKPOINT" -k $KMER --model-output "$OUTPUT/finetuning/$MODEL/supervised_model.pth"
        else
            python supervised_learning.py --input_path "$DATA/res101.mat" --model "$MODEL" --output_dir "$OUTPUT/finetuning/$MODEL" --n_epoch 12 -k $KMER --model-output "$OUTPUT/finetuning/$MODEL/supervised_model.pth"
        fi
    fi
    HP_OUTPUT="$OUTPUT/results/bzsl_output_${MODEL}_finetuned.json"

# run model without finetuning
else
    EMBEDDINGS="$OUTPUT/embeddings/dna_embeddings_$MODEL.csv"
    mkdir -p "$OUTPUT/embeddings"
    cd $ROOT/DNA_Embeddings
    if [ -n "$CHECKPOINT" ]; then
        python bert_extract_dna_feature.py --input_path "$DATA/res101.mat" --model "$MODEL" --checkpoint "$CHECKPOINT" --output "$EMBEDDINGS" -k "$KMER"
    else
        python bert_extract_dna_feature.py --input_path "$DATA/res101.mat" --model "$MODEL" --output "$EMBEDDINGS" -k "$KMER"
    fi
    HP_OUTPUT="$OUTPUT/results/bzsl_output_$MODEL.json"
fi

# BZSL tuning
cd $ROOT/BZSL-Python
mkdir -p "$OUTPUT/results"
python Demo.py --datapath "$DATA" --embeddings "$EMBEDDINGS" --tuning --output "$HP_OUTPUT"
