import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Set the font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

train = pd.read_csv("../data/supervised_train.csv", sep=None)
test = pd.read_csv("../data/supervised_test.csv", sep=None)

target_level = "species_name"

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


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


model = CNNModel(1, 1653)
model.load_state_dict(torch.load("../model_checkpoints/new_model_CNN.pth"))
model.to(device)
print("Model Loaded Succesfully")

# Getting the DNA embeddings of all data from the last dense layer
model2 = nn.Sequential(*list(model.children())[:-1])
model2.eval()


def data_from_df(df, target_level):
    barcodes = df["nucleotides"].to_list()
    species = df[target_level].to_list()
    orders = df["order_name"].to_list()

    print(len(barcodes), len(species))
    # Number of training samples and entire data
    N = len(barcodes)

    # Reading barcodes and labels into python list
    labels = []
    species_idx = sorted(set(species))

    for i in range(N):
        if len(barcodes[i]) > 0:
            barcodes.append(barcodes[i])
            # labels.append(species[i])
            labels.append(species_idx.index(species[i]))

    sl = 660
    nucleotide_dict = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    X = np.zeros((N, sl, 5), dtype=np.float32)
    for i in range(N):
        for j in range(sl):
            if len(barcodes[i]) > j:
                k = nucleotide_dict[barcodes[i][j]]
                X[i][j][k] = 1.0

    print(
        X.shape,
    )
    return X, np.array(labels), orders


X_test, y_test, orders = data_from_df(test, target_level)
X, y, train_orders = data_from_df(train, target_level)

dna_embeddings = []

with torch.no_grad():
    for i in range(X_test.shape[0]):
        inputs = torch.tensor(X_test[i]).view(-1, 1, 660, 5).to(device)
        dna_embeddings.extend(model2(inputs).cpu().numpy())


train_embeddings = []

with torch.no_grad():
    for i in range(X.shape[0]):
        inputs = torch.tensor(X[i]).view(-1, 1, 660, 5).to(device)
        train_embeddings.extend(model2(inputs).cpu().numpy())


print(f"There are {len(dna_embeddings)} points in the dataset")
latent = np.array(dna_embeddings).reshape(-1, 500)
print(latent.shape)
train_latent = np.array(train_embeddings).reshape(-1, 500)


from sklearn.linear_model import Perceptron

print("training the classifier")
clf = Perceptron(tol=1e-3, random_state=0, verbose=False, early_stopping=True, max_iter=100).fit(train_latent, y)
print(f"Test Accuracy: {clf.score(latent, y_test)}")
