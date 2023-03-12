import argparse
import os
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import pickle
from tqdm import tqdm
from os.path import dirname, abspath
project_dir = dirname(dirname(abspath(__file__)))
sys.path.append(project_dir)
from util.model import BERT
from util.dataset import SampleDNAData



def init_dataset_and_loader(args):
    dataset = SampleDNAData(file_path=args.data_path, k_mer=args.k_mer, data_count=args.data_count,
                            max_mask_count=args.max_mask_count, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset, dataloader


def init_model(args, dataset, device):
    model = BERT(dataset.vocab_size, args.d_model, dataset.max_len, 2, args.n_layers, 32, 32,
                 args.n_heads, device=device)
    return model


def train(args, dataloader, model, device):
    epoch_loss_list = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # strat training
    for epoch in range(args.continue_epoch, args.training_epoch + 1):
        epoch_loss = 0
        pbar = tqdm(dataloader)
        for ids, seg, msk_pos, masked_tokens, is_pos in pbar:
            pbar.set_description("Epoch: " + str(epoch))
            ids = ids.to(device)
            seg = seg.to(device)
            msk_pos = msk_pos.to(device)
            masked_tokens = masked_tokens.to(device)
            is_pos = is_pos.to(device)

            optimizer.zero_grad()
            logits_lm, logits_clsf, outputs = model(ids, seg, msk_pos)

            loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
            loss_lm = (loss_lm.float()).mean()
            loss_clsf = criterion(logits_clsf, torch.squeeze(is_pos))  # for sentence classification
            loss = loss_lm + loss_clsf

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss_list.append(epoch_loss)

        print(f"epoch {epoch}: Loss is {epoch_loss}")

        # every 50 epoch save the checkpoints and save the loss in a list
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(args.saving_path, "model_" + str(epoch) + '.pth'))
            torch.save(optimizer.state_dict(), args.saving_path + "optimizer_" + str(epoch) + '.pth')

            a_file = open(args.saving_path + "loss.pkl", "wb")
            pickle.dump(epoch_loss_list, a_file)
            a_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help="path to the input tsv.")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--k_mer', type=int, default=4, help="k for k-mer.")

    parser.add_argument('--data_count', type=int, default=256,
                        help="data_count")  # TODO: figure out what this is. write better explanation.
    parser.add_argument('--max_mask_count', type=int, default=80,
                        help="max_mask_count")  # TODO: figure out what this is. write better explanation.
    parser.add_argument('--max_len', type=int, default=512,
                        help="input sequence length.")  # TODO: figure out what this is. write better explanation.

    parser.add_argument('--mask_ratio', type=float, default=0.15,
                        help="mask ratio for masked language model")
    parser.add_argument('--training_epoch', type=int, default=1000,
                        help="number of epoch.")
    parser.add_argument('--continue_epoch', type=int, default=0,
                        help="epoch we want to continue training on.")  # TODO: continue training is not implement, for now.
    parser.add_argument('--d_model', type=int, default=768, help="number of hidden dimensions of BERT.")
    parser.add_argument('--n_heads', type=int, default=12, help="number of attention heads.")
    parser.add_argument('--n_layers', type=int, default=12, help="number of transformer blocks.")

    args = parser.parse_args()

    args.saving_path = os.path.join("outputs", "mask_ratio_" + str(args.mask_ratio).replace('.', '_'),
                                    "k_mer_" + str(args.k_mer))
    os.makedirs(args.saving_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloader to get a batch of data
    dataset, dataloader = init_dataset_and_loader(args=args)

    model = init_model(args, dataset, device)

    train(args, dataloader, model, device)
