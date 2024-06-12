import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class Bert_With_Prediction_Head(nn.Module):
    def __init__(self, out_feature, bert_model, dim=768, embedding_dim=768, model_type: str = "bioscanbert"):
        super().__init__()

        self.bert_model = bert_model
        self.model_type = model_type

        self.lin1 = nn.Linear(dim, embedding_dim)
        self.lin2 = nn.Linear(embedding_dim, out_feature)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if self.model_type == "dnabert2":
            x = self.bert_model(x)[0][0].mean(dim=0)
        else:
            x = self.bert_model(x).hidden_states[-1].mean(dim=1)

        x = self.tanh(self.dropout(self.lin1(x)))
        feature = x
        x = self.lin2(x)
        return x, feature

    def load_bert_model(self, path_to_ckpt):
        state_dict = torch.load(path_to_ckpt, map_location=torch.device("cpu"))
        state_dict = remove_extra_pre_fix(state_dict)
        self.bert_model.load_state_dict(state_dict)


def categorical_cross_entropy(outputs, target, num_classes=1653):
    m = nn.Softmax(dim=1)
    pred_label = torch.log(m(outputs))
    target_label = F.one_hot(target, num_classes=num_classes)

    loss = (-pred_label * target_label).sum(dim=1).mean()
    return loss


def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    return new_state_dict


def train_and_eval(model, trainloader, testloader, device, lr=0.0025, n_epoch=12):
    # criterion = nn.CrossEntropyLoss()
    # criterion = categorical_cross_entropy()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    print("start training")
    loss = None
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader))
        for _i_batch, (inputs, labels) in pbar:
            if loss is not None:
                pbar.set_description("Epoch: " + str(epoch) + " || loss: " + str(loss.item()))
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, _ = model(inputs)

            # loss = criterion(outputs, labels)

            loss = categorical_cross_entropy(outputs, labels)

            loss.backward()

            optimizer.step()
            # print statistics
            running_loss += loss.item()

        with torch.no_grad():
            model.eval()
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

        print(
            "Epoch: "
            + str(epoch)
            + "|| loss: "
            + str(running_loss / len(trainloader))
            + "|| Accuracy: "
            + str(train_correct / train_total)
            + "|| Val Accuracy: "
            + str(correct / total)
            + "|| lr: "
            + str(scheduler.get_last_lr())
        )
        running_loss = 0
        scheduler.step()

    print("Finished Training")
