import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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


def data_from_df(df, target_level):
    barcodes = df["nucleotides"].to_list()
    species = df[target_level].to_list()
    orders = test["order_name"].to_list()

    print(len(barcodes), len(species))
    # Number of training samples and entire data
    N = len(barcodes)

    # Reading barcodes and labels into python list
    labels = []
    species_idx = sorted(set(species))

    for i in range(N):
        if len(barcodes[i]) > 0:
            barcodes.append(barcodes[i])
            labels.append(species_idx.index(species[i]))

    sl = 660
    nucleotide_dict = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    X = np.zeros((N, sl, 5), dtype=np.float32)
    for i in range(N):
        for j in range(sl):
            if len(barcodes[i]) > j:
                k = nucleotide_dict[barcodes[i][j]]
                X[i][j][k] = 1.0

    # print(X.shape, )
    return X, np.array(labels), orders


X_train, y_train, train_orders = data_from_df(train, target_level)
X_test, y_test, orders = data_from_df(test, target_level)

numClasses = max(y_train) + 1
print(numClasses)

model = CNNModel(1, numClasses)
model.to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Convert the data to PyTorch tensors and create DataLoader if needed
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

log_interval = 200

# Training
for epoch in range(20):
    epoch_start_time = time.time()
    model.train()
    total_acc, total_count = 0, 0
    start_time = time.time()
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.view(-1, 1, 660, 5).to(device), labels.to(device)

        optimizer.zero_grad()
        predicted_label = model(inputs)
        # print(predicted_label.shape)
        loss = criterion(predicted_label, labels)
        loss.backward()
        optimizer.step()

        total_acc += (predicted_label.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc / total_count)
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.view(-1, 1, 660, 5).to(device), labels.to(device)
            predicted_label = model(inputs)
            loss = criterion(predicted_label, labels)
            total_acc += (predicted_label.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    accu_val = total_acc / total_count
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "Test accuracy {:8.3f} ".format(epoch, time.time() - epoch_start_time, accu_val)
    )
    print("-" * 59)

torch.save(model.state_dict(), "../model_checkpoints/new_model_CNN.pth")
