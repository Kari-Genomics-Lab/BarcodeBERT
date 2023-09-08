from bayesian_model import Model
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--side_info', default='original',type=str)
parser.add_argument('--pca_dim', default=500, type=int)
parser.add_argument('--datapath', default='../data/', type=str)
parser.add_argument('--tuning', default=False, type=lambda x: (str(x).lower()=='true'))
parser.add_argument('--alignment', default=True, type=lambda x: (str(x).lower()=='true'))


"""
You may alter the model hyperparameters -- k_0, k_1, m, s, K -- inside train_and_eval() function from Model class.
This setting will reproduce the results presented in the paper 
"""

args = parser.parse_args()

model = Model(args)
model.train_and_eval()