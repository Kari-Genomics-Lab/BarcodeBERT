import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
import umap

test = pd.read_csv("data/unseen.tsv", sep='\t')

target_level='species_name'

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# CNN model architecture
class CNNModel(nn.Module):
    def __init__(self, n_input, n_output):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(n_input, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 1))
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 1))
        self.flatten = nn.Flatten()
        self.bn4 = nn.BatchNorm1d(1920)
        self.dense1 = nn.Linear(1920, 500)
        self.dense2 = nn.Linear(500, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.bn3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.bn4(x)
        x = nn.Tanh()(self.dense1(x))
        x = self.dense2(x)
        return x

model = CNNModel(1, 1390)
model.load_state_dict(torch.load('model_checkpoints/model_CNN.pth'))
model.to(device)
print('Model Loaded Succesfully')

# Getting the DNA embeddings of all data from the last dense layer
model2 = nn.Sequential(*list(model.children())[:-1])
model2.eval()

def data_from_df(df, target_level):
    barcodes =  df['nucleotides'].to_list()
    species  =  df[target_level].to_list()
    print(len(barcodes), len(species))
    # Number of training samples and entire data
    N = len(barcodes)

    # Reading barcodes and labels into python list
    labels=[]
    species_idx=sorted(list(set(species)))

    for i in range(N):
        if len(barcodes[i])>0:
            barcodes.append(barcodes[i])
            labels.append(species_idx.index(species[i]))

    sl = 660
    nucleotide_dict = {'A': 0, 'C':1, 'G':2, 'T':3, 'N':4}
    X=np.zeros((N,sl,5), dtype=np.float32)
    for i in range(N):
        Nt=len(barcodes[i])

        for j in range(sl):
            if(len(barcodes[i])>j):
                k=nucleotide_dict[barcodes[i][j]]
                X[i][j][k]=1.0

    print(X.shape, )
    return X, np.array(labels)

X_unseen, y_unseen = data_from_df(test, target_level)
dna_embeddings = []

with torch.no_grad():
    for i in range(X_unseen.shape[0]):
        inputs = torch.tensor(X_unseen[i]).view(-1, 1, 660, 5).to(device)
        dna_embeddings.extend(model2(inputs).cpu().numpy())
    
print(f"There are {len(dna_embeddings)} points in the dataset")
latent = np.array(dna_embeddings).reshape(-1,500)
print(latent.shape)
embedding = umap.UMAP(random_state=42).fit_transform(latent)

import matplotlib.pyplot as plt 
fig, ax = plt.subplots(nrows=1, ncols=1) 
ax.set_title("Representation of the Latent Space")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")

ax.scatter(embedding[:, 0],
           embedding[:, 1],
           c=y_unseen,
           s=1,
           cmap='Spectral')
plt.savefig('1D_CNN_embeddings.png',dpi=150)

## Histogram of Nearest Neighbors
metrics = ['manhattan', 'cosine', 'minkowski']
scores = []
neighbour_size = 1
gt = y_unseen
for metric in metrics:
    nbrs = NearestNeighbors(n_neighbors=neighbour_size+1, metric=metric).fit(embedding)
    _, neighbour_indices = nbrs.kneighbors(embedding)
    neighbour_indices = neighbour_indices.astype(np.int32)[:,1:]
    #print(neighbour_indices)
    neighbour_indices = neighbour_indices.reshape(-1)
    #print(neighbour_indices)
    
    
    gt_indices = np.array([[idx]*neighbour_size for idx in range(len(gt))]).reshape(-1).astype(np.int32)
    #print(gt_indices.shape)

    gt = np.array(gt)

    neighbor_gt = gt[neighbour_indices]
    #print(np.sum(gt[gt_indices] == gt[neighbour_indices]))

    cluster_distribution = pd.Series(gt[gt_indices]).value_counts().to_dict()
    #print(cluster_distribution)
    good_neighbours = pd.Series( gt[gt_indices][gt[gt_indices] == gt[neighbour_indices]]).value_counts().to_dict()
    #print(good_neighbours)
    score = 0
    n_clusters = 0
    for cluster in cluster_distribution:
        score += good_neighbours[cluster]/cluster_distribution[cluster]
        n_clusters += 1
    scores.append(score/n_clusters)

print(scores)
print("Best Metric: ", metrics[scores.index(max(scores))], "Accuracy: ", max(scores))
best_metric = metrics[scores.index(max(scores))]