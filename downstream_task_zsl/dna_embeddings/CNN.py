import torch.nn as nn
import torch.nn.functional as F
from opt_einsum.backends import torch
from torch import optim
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import datetime
import os


class Model(nn.Module):
    def __init__(self, in_feature, out_feature, dim=1840, embedding_dim=500):
        super().__init__()
        self.pool = nn.MaxPool2d((3, 1))
        self.conv1 = nn.Conv2d(in_channels=in_feature, out_channels=64, kernel_size=(3, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(16)
        self.flat = nn.Flatten(1, 3)
        self.lin1 = nn.Linear(dim, embedding_dim)
        self.lin2 = nn.Linear(embedding_dim, out_feature)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dropout(self.conv1(x))
        x = self.pool(self.bn1(F.relu(x)))
        x = self.pool(self.bn2(F.relu(self.dropout(self.conv2(x)))))
        x = self.pool(self.bn3(F.relu(self.dropout(self.conv3(x)))))
        x = self.flat(x)
        x = self.tanh(self.dropout(self.lin1(x)))
        feature = x
        x = self.lin2(x)
        return x, feature


# def categorical_cross_entropy(outputs, target):
#     m = nn.Softmax(dim=1)
#     loss = nn.NLLLoss()(torch.log(m(outputs)), target)
#     # print(torch.log(m(outputs)).shape)
#     # print(target.shape)
#     # print(loss)
#     return loss

def categorical_cross_entropy(outputs, target, num_classes):
    m = nn.Softmax(dim=1)
    pred_label = torch.log(m(outputs))
    target_label = F.one_hot(target, num_classes=num_classes)
    loss = (-pred_label * target_label).sum(dim=1).mean()
    return loss


def train_and_eval(args, model, trainloader, testloader, device, lr=0.005, n_epoch=12, num_classes=1213):
    criterion = nn.CrossEntropyLoss()
    # criterion = categorical_cross_entropy()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    print('start training')
    epoch_pbar = tqdm(range(n_epoch))
    for epoch in epoch_pbar:  # loop over the dataset multiple times
        epoch_pbar.set_description('Epoch: ' + str(epoch))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, _ = model(inputs)
            # print(outputs[0])
            # print(labels.shape)
            # exit()
            # loss = criterion(outputs, labels)
            loss = categorical_cross_entropy(outputs, labels, num_classes)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:
                with torch.no_grad():
                    train_correct = 0
                    train_total = 0
                    for data in trainloader:
                        inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
                        labels = labels.int()

                        # calculate outputs by running images through the network
                        outputs, _ = model(inputs)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        predicted = predicted.int()
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()

                    correct = 0
                    total = 0
                    for data in testloader:
                        inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
                        labels = labels.int()
                        # calculate outputs by running images through the network
                        outputs, _ = model(inputs)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        predicted = predicted.int()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                if i > 1:
                    print("Epoch: " + str(epoch) + " ||Iteration: " + str(i) + "|| loss: " + str(
                        running_loss / 100) + "|| Accuracy: " + str(
                        train_correct / train_total) + "|| Val Accuracy: " + str(correct / total) + "|| lr: " + str(scheduler.get_last_lr()))
                running_loss = 0
        scheduler.step()
    
    print('Finished Training')
    if args.save_model:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
        save_dir = os.path.join(args.model_output_dir, args.taxonomy_level, formatted_datetime)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model.pth')
        torch.save(model.state_dict(), save_path)
        dict_args = vars(args)
        with open(os.path.join(save_dir, 'argparse_config.txt'), 'w') as f:
            for key in dict_args:
                 f.write(f'{key} \t -> {dict_args[key]}\n')
        
        
        print("Model is saved in: '" + save_path + "'")
        
        