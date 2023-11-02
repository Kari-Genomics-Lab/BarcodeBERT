# BarcodeBERT

A pre-trained representation from a transformer model for inference on insect DNA barcoding data.

### Reproducing the results from the paper

0. Clone this repository and install the required libraries

1. Download the [data](https://github.com/jerryji1993/DNABERT)

##### CNN model
Training: 
```
cd paper/CNN
python supervised_learning.py
```

Evaluation:
```
python genus_1NN.py
python Linear_probing.py
```

##### BarcodeBERT

Model Pretraining:
```
cd /paper/BarcodeBERT
pip install --no-index -r requirements.txt
python MGPU_MLM_train.py --input_path=../../data/pre_training.tsv --k_mer=4 --stride=4
python MGPU_MLM_train.py --input_path=../../data/pre_training.tsv --k_mer=5 --stride=5
python MGPU_MLM_train.py --input_path=../../data/pre_training.tsv --k_mer=6 --stride=6
```

Evaluation:
```
python MLM_genus_test.py 4
python MLM_genus_test.py 5
python MLM_genus_test.py 6

python Linear_probing.py 4
python Linear_probing.py 5
python Linear_probing.py 6
```

Model Fine-tuning
The input_path for fine-tuning the model should be a path to a folder that has train, test, and dev file.
```
python fine_tuning.py --input_path= path_to_the_input_folder --Pretrained_checkpoint_path path_to_the_pretrained_model  --k_mer=4 --stride=4
python fine_tuning.py --input_path= path_to_the_input_folder --Pretrained_checkpoint_path path_to_the_pretrained_model  --k_mer=5 --stride=5
python fine_tuning.py --input_path= path_to_the_input_folder --Pretrained_checkpoint_path path_to_the_pretrained_model  --k_mer=6 --stride=6
```


##### DNABERT
To fine-tune the model on our data, you first need to follow the instructions in the [DNABERT repository](https://github.com/jerryji1993/DNABERT) original repository to donwnload the model weights. Place them in the `dnabert` folder and then run the following:

```
cd paper/DNABERT
pip install --no-index -r requirements.txt
python supervised_learning.py --input_path=../../data -k 4 --model dnabert --checkpoint dnabert/4-new-12w-0
python supervised_learning.py --input_path=../../data -k 6 --model dnabert --checkpoint dnabert/6-new-12w-0
python supervised_learning.py --input_path=../../data -k 5 --model dnabert --checkpoint dnabert/5-new-12w-0
```

Evaluation:


###### DNABERT-2

To fine-tune the model on our dataset, you need to follow the instructions in [DNABERT2 repository](https://github.com/Zhihan1996/DNABERT_2) for fine-tuning the model on new dataset. You can use the same input path that is used for fine-tuning BarcodeBERT as the input path to DNABERT2. 

Evaluation:
```

```
<!--- 

### Using BarcodeBERT as feature extractor in your own biodiversity analysis:

0. Clone this repository and install the required libraries

1. Download the pre-trained weights

2. Produce the features
**Note**: The model is ready to be used on data directly downloaded from BOLD. To use the model on your own data, please format the .tsv input file accordingly. 


### Fine-Tuning BarcodeBERT using your own data

0. Clone this repository and install the required libraries

1. Download the pre-trained weights

2. Fine-Tune the model
**Note**: The model is ready to be used on data directly downloaded from BOLD. To use the model on your own data, please format the .tsv input file accordingly. 

3. Test the fine-tuned model on the test dataset.






0. Download the [data](https://vault.cs.uwaterloo.ca/s/YojSrfn7n2iLfa9)
1. Make sure you have all the required libraries before running (remove the --no-index flags if you are not training on CC)

```
pip install -r requirements.txt
```

--!>
