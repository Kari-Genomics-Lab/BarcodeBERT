export KMER=6
export SOURCE='/home/pmillana/projects/def-khill22/pmillana/DNABERT/'
export MODEL_PATH=$SOURCE/examples/ft/$KMER/
export DATA_PATH=$SOURCE/examples/sample_data/ft/$KMER
export PREDICTION_PATH=$SOURCE/examples/result/$KMER

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=$SOURCE/vocab \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 512 \
    --per_gpu_pred_batch_size=128   \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 40