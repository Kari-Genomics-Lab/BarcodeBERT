# DNA embedding extraction

## Usage

### Data Download

The INSECT data used for these scripts can be found at /project/3dlg-hcvc/bioscan/BZSL_INSECT_data.zip.

### Embedding extraction

To generate DNA embeddings on the INSECT dataset, run a command similar to one of the below:

```
# BIOSCAN BERT model
python bert_extract_dna_feature.py --input_path ../../data/INSECT/res101.mat --model bioscanbert --checkpoint ../../data/bioscanbert/latest_model_5mer.pth --output ../../data/INSECT/dna_embedding_insect_bioscanbert_new.csv -k 5

# DNABERT
python bert_extract_dna_feature.py --model dnabert --checkpoint ../data/dnabert_pretrained --output ../data/INSECT/dna_embedding_insect_dnabert.csv

# DNABERT-2
python bert_extract_dna_feature.py --model dnabert2 --output ../data/INSECT/dna_embedding_insect_dnabert2.csv
```

Note that for DNABERT-2, I ran into some issues with `trans_b` no longer being a supported parameter for `tl.dot`, and
in the end just disabled flash_attention in the repo by setting `flash_attn_qkvpacked_func` to `None` in bert_layers.py.
However, the code which needed to be modified is from huggingface and not part of this repository, so you may need to make
that change manually yourself. The alternative is downgrading our python and pytorch versions until the particular version
of triton we need is supported.

#### Model weights

- BIOSCAN BERT: model saved in BIOSCAN google drive folder
- DNABERT: downloadable from [DNABERT repository](https://github.com/jerryji1993/DNABERT)
- DNABERT-2: provided in Huggingface

### Fine-tuning

To fine tune the model (Pablo's BERT or DNABERT), run the following:
```
# Pablo's BERT
python supervised_learning.py --input_path path/to/res101.mat --model bioscanbert --output_dir path/to/output/ --n_epoch 12

# DNABERT
python supervised_learning.py --input_path path/to/res101.mat --model dnabert --output_dir path/to/output/ --n_epoch 12
```

For DNABERT-2, you will need to use the [DNABERT-2 repository](https://github.com/Zhihan1996/DNABERT_2) and apply 
fine-tuning with the data files (`train.csv` and `dev.csv`) created at 
`/project/3dlg-hcvc/bioscan/bzsl/dnabert2_fine_tuning/`.
