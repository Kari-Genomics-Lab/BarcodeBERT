# DNA language models for taxonomic classification

In this branch we pre-train BarcodeBERT on the BIOSCAN-5M dataset and compare its performance on the following tasks:
* Classification of DNA barcodes from seen species.
* Classification of barcodes from unseen species from seen genera.

Currently, we only support the following encoders:
* DNABERT-S
* DNABERT-2
* HyenaDNA
* The nucleotide transformer (NT)
* BarcodeBERT

| Model        | Architecture | SSL-Pretraining | Tokens seen   |  **Seen: Species**  (Fine-tuned )    | Linear probe  **Seen: Species** (Linear Probe) | **Unseen: Genus** (1NN-Probe) |
 -------------|--------------|-----------------|---------------|:-----------------:|:-------------:|:----------------:|  
 BLAST*       | --           | --              | --            | **99.78**     |     ---       |  **58.74**  
 CNN baseline | CNN          | --              | --            | 97.70             | --            | 29.88
 NT           | Transformer  | Multi-Species   | 300\,B        | 98.99             | 52.41         | 21.67  
 DNABERT-2    | Transformer  | Multi-Species   | 512\,B        | **99.23**       | 67.81         | 17.99  
 DNABERT-S    | Transformer  | Multi-Species   | ~1,000\,B     | 98.99             | **95.50**        | 17.70  
 HyenaDNA     | SSM          | Human DNA       | 5\,B          | 98.71             | 54.82         | 19.26  
 BarcodeBERT  | Transformer  | DNA barcodes    | 5\,B          | 98.52             | 91.93         | 23.15  
 Ours (8-4-4) | Transformer  | DNA barcodes    | 7\,B          | **99.28**       | 94.47         | **47.03**  
Our pre-trained transformer model on DNA barcodes follows the implementation in the BarcodeBERT [paper](https://arxiv.org/abs/2311.02401)  for inference on insect DNA barcoding data.

### Reproducing the results from the paper

0. Clone this repository and install the required libraries by running
```shell
pip install -e .
```

1. Download the [metadata file](https://drive.google.com/drive/u/1/folders/1TLVw0P4MT_5lPrgjMCMREiP8KW-V4nTb) and copy it into the data folder

2. Split the metadata file into smaller file according to the different partitionss
```shell
cd data/
python data_split.py BIOSCAN-5M_Dataset_metadata.tsv
```

3. Improved BarcodeBERT model pipeline

```bash
python barcodebert/pretraining.py --data_path=data/pretraining.csv --k_mer=8 --n_layers=4 --n_heads=4 --data_dir=data/
python barcodebert/knn_probing.py --pretrained-checkpoint=path/to/checkpoint_pretraining.pt
python barcodebert/finetuning.py --pretrained-checkpoint=path/to/checkpoint_pretraining.pt
python barcodebert/finetuning.py --pretrained-checkpoint=path/to/checkpoint_pretraining.pt --freeze-encoder
```

4. Baseline model pipeline
```bash
python baselines/knn_probing.py --backbone=<DESIRED-BACKBONE>
python baselines/linear_probing.py --backbone=<DESIRED-BACKBONE>
python baselines/finetuning.py --backbone=<DESIRED-BACKBONE> --data-dir=data/ --batch_size=32
```

5. BLAST
```shell
cd data/
python to_fasta.py --input_file=supervised_train.csv &&
python to_fasta.py --input_file=supervised_test.csv &&
python to_fasta.py --input_file=unseen.csv

makeblastdb -in supervised_train.fas -title train -dbtype nucl -out train.fas
blastn -query supervised_test.fas -db train.fas -out results_supervised_test.tsv -outfmt 6 -num_threads 16
blastn -query unseen.fas -db train.fas -out results_unseen.tsv -outfmt 6 -num_threads 16
```


## Citation

If you find BarcodeBERT useful in your research please consider citing:

    @misc{arias2023barcodebert,
      title={{BarcodeBERT}: Transformers for Biodiversity Analysis},
      author={Pablo Millan Arias
        and Niousha Sadjadi
        and Monireh Safari
        and ZeMing Gong
        and Austin T. Wang
        and Scott C. Lowe
        and Joakim Bruslund Haurum
        and Iuliia Zarubiieva
        and Dirk Steinke
        and Lila Kari
        and Angel X. Chang
        and Graham W. Taylor
      },
      year={2023},
      eprint={2311.02401},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      doi={10.48550/arxiv.2311.02401},
    }




## Development

Our code is automatically standardized using [pre-commit](https://pre-commit.com/).

When developing the codebase, please install pre-commit and our pre-commit hooks with the following code:

```bash
pip install -e .[dev]
pre-commit install
```

This will then automatically change your code style to the [black](https://github.com/psf/black) format when you try to commit it, and catch any [flake8](https://flake8.pycqa.org/) errors.
If there are any corrections automatically made by pre-commit or corrections you need to implement, the commit will not initially go through until you stage the appropriate changes and try to commit again.


<!---

### Using BarcodeBERT as feature extractor in your own biodiversity analysis:

0. Clone this repository and install the required libraries

1. Download the pre-trained weights

2. Produce the features
**Note**: The model is ready to be used on data directly downloaded from BOLD. To use the model on your own data, please format the .csv input file accordingly.


### Fine-Tuning BarcodeBERT using your own data

0. Clone this repository and install the required libraries

1. Download the pre-trained weights

2. Fine-Tune the model
**Note**: The model is ready to be used on data directly downloaded from BOLD. To use the model on your own data, please format the .csv input file accordingly.

3. Test the fine-tuned model on the test dataset.






0. Download the [data](https://vault.cs.uwaterloo.ca/s/YojSrfn7n2iLfa9)
1. Make sure you have all the required libraries before running (remove the --no-index flags if you are not training on CC)

```
pip install -r requirements.txt
```

--!>
