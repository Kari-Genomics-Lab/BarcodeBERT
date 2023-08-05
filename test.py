import os
import sys
import argparse
from transformers import BertForPreTraining
from transformers import BertConfig
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from util.dataset import SampleDNAData
from tqdm import tqdm
from collections import OrderedDict

def test(args, dataloader, device, model):
    criterion = nn.CrossEntropyLoss()

    totalloss = 0
    steps = len(dataloader)
    pbar_iter_level = tqdm(total=steps)
    for step, batch in enumerate(dataloader):
            ids = batch[0].to(device)  # 'tokens'
            seg = batch[1].to(device)  # 'segment'
            mask = batch[2].to(device)
            msk_pos = batch[3].to(device)  # 'msk_pos'
            masked_tokens = batch[4].to(device)  # 'msk_tok'
            is_pos = batch[5].to(device)  # 'label'

            inputs = {
                'input_ids': ids,
                'token_type_ids': seg,
                'attention_mask': mask
            }

            with torch.no_grad():
                outputs = model(**inputs)

                msk_pos = msk_pos[:, :, None].expand(-1, -1, outputs['prediction_logits'].size(-1))
                masked_prediction_logits = torch.gather(outputs['prediction_logits'], 1, msk_pos)

                seq_relationship_loss = criterion(outputs['seq_relationship_logits'], torch.squeeze(is_pos))

                prediction_loss = criterion(masked_prediction_logits.transpose(1, 2), masked_tokens)

                w = args['loss_weight']  # Weight for the importance of the masked language model
                loss = w * prediction_loss + (1 - w) * seq_relationship_loss

                totalloss += loss.item()

                pbar_iter_level.set_description("Loss: " + str(loss.item()))
                pbar_iter_level.update(1)
    
    return totalloss / steps

def prepare(dataset, batch_size=32, pin_memory=False, num_workers=0):

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False)

    return dataloader

def ddp_state_dict_to_regular(state_dict, offset):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[offset:] # remove 'module.bert'
        new_state_dict[name] = v
    return new_state_dict
    

def main(args):
    sys.stdout.write("Loading the dataset is started.\n")
    args['input_path'] = os.path.join(args['input_dir'], args['name_of_dataset'] + ".tsv")

    dataset = SampleDNAData(file_path=args['input_path'], k_mer=args['k_mer'], max_mask_count=args['max_mask_count'],
                            max_len=args['max_len'])
    
    sys.stdout.write("loading the model.\n")
    vocab_size = 5 ** args['k_mer'] + 4  # '[PAD]', '[CLS]', '[SEP]',  '[MASK]'

    configuration = BertConfig(vocab_size=vocab_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initializing a model (with random weights) from the bert-base-uncased style configuration

    model = BertForPreTraining(configuration).to(device)

    state_dict = ddp_state_dict_to_regular(torch.load(args['checkpoint_path']), 7)
    model.load_state_dict(state_dict)

    sys.stdout.write("Model is loaded.\n")

    dataloader = prepare(dataset, batch_size=args['batch_size'])

    averge_loss = test(args, dataloader, device, model)
    sys.stdout.write("Averge loss: {}\n".format(averge_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', action='store', type=str)
    parser.add_argument('--name_of_dataset', action='store', type=str)
    
    parser.add_argument('--checkpoint_path', action='store', type=str)
    parser.add_argument('--k_mer', action='store', type=int, default=4)
    parser.add_argument('--max_mask_count', action='store', type=int, default=80)
    parser.add_argument('--max_len',action='store', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--loss_weight', action='store', type=float, default=0.5)

    args = vars(parser.parse_args())

    sys.stdout.write("\Testing Parameters:\n")
    for key in args:
        sys.stdout.write(f'{key} \t -> {args[key]}\n')

    main(args)