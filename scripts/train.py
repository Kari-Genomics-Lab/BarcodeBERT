import os
import sys
import argparse
from transformers import BertForPreTraining
from transformers import BertConfig
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from util.dataset import SampleDNAData
from tqdm import tqdm
import wandb

def train(args, dataloader, device, model, optimizer):
    # start training
    criterion = nn.CrossEntropyLoss()

    epoch_loss_list = []
    training_epoch = args['epoch']
    continue_epoch = 0

    saving_path = "model_checkpoints/"
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    if args['checkpoint']:
        continue_epoch = 0
        model.load_state_dict(torch.load(saving_path + f'model_{continue_epoch}.pth'))
        optimizer.load_state_dict(torch.load(saving_path + f"optimizer_{continue_epoch}.pth"))
        continue_epoch += 1
        a_file = open(saving_path + "loss.pkl", "rb")
        epoch_loss_list = pickle.load(a_file)
        print("Trainig is countinue...")

    sys.stdout.write("Training is started:\n")

    if args['activate_wandb']:
        wandb.init(project="BioScan_transformer", name="Init_run")

    steps_per_epoch = len(dataloader)
    pbar_epoch_level = tqdm(range(continue_epoch, training_epoch + 1))
    for epoch in pbar_epoch_level:
        pbar_epoch_level.set_description("Epoch: " + str(epoch))
        epoch_loss = 0
        dataloader.sampler.set_epoch(epoch)
        pbar_iter_level = tqdm(total=steps_per_epoch)
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

            optimizer.zero_grad()
            outputs = model(**inputs)

            msk_pos = msk_pos[:, :, None].expand(-1, -1, outputs['prediction_logits'].size(-1))
            masked_prediction_logits = torch.gather(outputs['prediction_logits'], 1, msk_pos)

            seq_relationship_loss = criterion(outputs['seq_relationship_logits'], torch.squeeze(is_pos))

            prediction_loss = criterion(masked_prediction_logits.transpose(1, 2), masked_tokens)

            w = args['loss_weight']  # Weight for the importance of the masked language model
            loss = w * prediction_loss + (1 - w) * seq_relationship_loss

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            if args['activate_wandb']:
                wandb.log({"loss": loss.item()}, step=step + epoch * steps_per_epoch)

                wandb.run.summary["epoch"] = epoch + 1
                # global_step = step + epoch * steps_per_epoch

                # if step == 0:
                #     wandb.log({f"custom_plot/xaxis": "epoch", "custom_plot/yaxis": "metric_value"}, step=global_step)

                wandb.log({}, commit=True)

            pbar_iter_level.set_description("Loss: " + str(loss.item()))
            pbar_iter_level.update(1)


        epoch_loss_list.append(epoch_loss)

        wandb.log({'epoch': epoch, 'epoch_loss': epoch_loss})
        wandb.log({}, commit=True)

        # every epoch save the checkpoints and save the loss in a list
        if epoch % 1 == 0:
            sys.stdout.write(f"Epoch {epoch} in device {device}: Loss is {epoch_loss}\n")
            torch.save(model.state_dict(), saving_path + "model_" + str(epoch) + '.pth')
            torch.save(optimizer.state_dict(), saving_path + "optimizer_" + str(epoch) + '.pth')

            a_file = open(saving_path + f"loss_{device}.pkl", "wb")
            pickle.dump(epoch_loss_list, a_file)
            a_file.close()


def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    # dataset = Your_Dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)

    return dataloader


def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)

    sys.stdout.write("Loading the dataset is started.\n")
    dataset = SampleDNAData(file_path=args['input_path'], k_mer=args['k_mer'], max_mask_count=args['max_mask_count'],
                            max_len=args['max_len'])

    sys.stdout.write("loading the model.\n")
    vocab_size = 5 ** args['k_mer'] + 4  # '[PAD]', '[CLS]', '[SEP]',  '[MASK]'
    # config = {
    #     "d_model": 768,
    #     "n_heads": 12,
    #     "n_layers": 12,
    #     "max_len": 512
    # }

    # model = VerySimpleModel(vocab_size, config["d_model"], args['max_len'], 2, config["n_layers"], 32, 32,
    #                     config["n_heads"], device=rank).to(rank)

    configuration = BertConfig(vocab_size=vocab_size)

    # Initializing a model (with random weights) from the bert-base-uncased style configuration

    model = BertForPreTraining(configuration).to(rank)

    # model = BERT(vocab_size, config["d_model"], args['max_len'], 2, config["n_layers"], 32, 32,
    #              config["n_heads"], device=rank)
    # model = SampleModel(vocab_size).to(rank)

    sys.stdout.write("Model is loaded.\n")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # dataloader = DataLoader(dataset, batch_size=args['batch_size'], pin_memory=False, shuffle=False,
    #                         sampler=DistributedSampler(dataset))
    dataloader = prepare(dataset, rank, world_size=world_size, batch_size=args['batch_size'])

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    train(args, dataloader, rank, model, optimizer)
    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', action='store', type=str)
    parser.add_argument('--checkpoint', action='store', type=bool, default=False)
    parser.add_argument('--k_mer', action='store', type=int, default=4)
    parser.add_argument('--max_mask_count', action='store', type=int, default=80)
    parser.add_argument('--max_len', action='store', type=int, default=512)
    parser.add_argument('--batch_size', action='store', type=int, default=8)
    parser.add_argument('--loss_weight', action='store', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--activate_wandb', default=False, action='store_true')

    args = vars(parser.parse_args())

    sys.stdout.write("\nTraining Parameters:\n")
    for key in args:
        sys.stdout.write(f'{key} \t -> {args[key]}\n')

    world_size = torch.cuda.device_count()

    mp.spawn(main, args=(world_size, args), nprocs=world_size)