import torch
import torch.nn as nn


class FinetuneBert(nn.Module):
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


def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    return new_state_dict
