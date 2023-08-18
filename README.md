# Bumblebee

A pre-trained representation from a transformers model for inference on insect DNA barcoding data.

Download the [data](https://vault.cs.uwaterloo.ca/s/YojSrfn7n2iLfa9)

# Set environment
For now, you can set the environment by typing
```shell
conda create -n Bumblebee python=3.10
conda activate Bumblebee
pip install -r requirements.txt

```
in the terminal.

# Pre-training
For single or multiple GPU pre-training, you can use the following command.
```shell
python train.py --input_dir {path to the data folder} --name_of_proj {name of the project you want to show in wandb} --name_of_dataset {name of the dataset without the suffix}
```
There are also other arguments you can use to set the training. Here is an example for pre-training with the small set: 
```shell
python train.py --input_dir data/for_training --name_of_proj Bioscan-transformer-small-dataset --name_of_dataset small_training --lr 0.00005  --betas_a 0.9 --betas_b 0.98 --eps 1e-06 --weight_decay 1e-05 --name_of_exp lr_5e-5_div_10 --name_of_run lr_5e-5_div_10 --epoch 50 --activate_wandb --activate_lr_scheduler --div_factor 10
```
Note: If you are using the solar for multi-GPU training, you may want to add `NCCL_P2P_LEVEL=NVL` in front of your command to enable P2P during the training.

# Test the pre-training
`TODO`

