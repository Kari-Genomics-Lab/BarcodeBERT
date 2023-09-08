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
import sys
print("User Current Version:-", sys.version)

def train(args, dataloader, device, model, optimizer, scheduler, saving_path):
    # start training
    sys.stdout.write("Enter train func:\n")
    criterion = nn.CrossEntropyLoss()

    epoch_loss_list = []
    training_epoch = args['epoch']
    continue_epoch = 0
    sys.stdout.write("Before optional loading:\n")
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
        wandb.init(project=args['name_of_proj'], name=args['name_of_run'])

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
            if args['activate_lr_scheduler']:
                scheduler.step()
            if args['activate_wandb']:

                # print()
                wandb.log({"loss": loss.item()}, step=step + epoch * steps_per_epoch)

                wandb.log({}, commit=True)

            pbar_iter_level.set_description("Loss: " + str(loss.item()))
            pbar_iter_level.update(1)

        epoch_loss_list.append(epoch_loss)

        if args['activate_wandb']:
            wandb.log({'epoch': epoch, 'epoch_loss': epoch_loss})
            wandb.log({}, commit=True)

        # every epoch save the checkpoints and save the loss in a list
        if epoch % 1 == 0:
            os.makedirs(saving_path, exist_ok=True)
            sys.stdout.write(f"Epoch {epoch} in device {device}: Loss is {epoch_loss}\n")
            torch.save(model.state_dict(), saving_path + "last_model" + '.pth')
            torch.save(optimizer.state_dict(), saving_path + "last_optimizer" + '.pth')

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


def main(rank: int, world_size: int, args, saving_path):
    torch.manual_seed(42)
    ddp_setup(rank, world_size)

    sys.stdout.write("Loading the dataset is started.\n")
    args['input_path'] = os.path.join(args['input_dir'], args['name_of_dataset'] + ".tsv")

    dataset = SampleDNAData(file_path=args['input_path'], k_mer=args['k_mer'], max_mask_count=args['max_mask_count'],
                            max_len=args['max_len'])

    sys.stdout.write("loading the model.\n")
    vocab_size = 5 ** args['k_mer'] + 4  # '[PAD]', '[CLS]', '[SEP]',  '[MASK]'
    configuration = BertConfig(vocab_size=vocab_size)

    # Initializing a model (with random weights) from the bert-base-uncased style configuration
    device = torch.device('cuda', rank)
    model = BertForPreTraining(configuration).to(device)

    sys.stdout.write("Model is loaded.\n")

    optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(args['betas_a'], args['betas_b']), eps=args['eps'], weight_decay=args['weight_decay'])

    dataloader = prepare(dataset, rank, world_size=world_size, batch_size=args['batch_size'])
    sys.stdout.write("optim and dataloader is ready\n")
    if args['activate_lr_scheduler']:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args['lr'],
                                                        steps_per_epoch=len(dataloader),
                                                        epochs=args['epoch'], div_factor=args['div_factor'])
        sys.stdout.write("scheduler is ready\n")
    else:
        scheduler = None

    model = DDP(model, device_ids=[rank], output_device=rank)
    # model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    sys.stdout.write("DDP is ready\n")

    sys.stdout.write("Model is ready.\n")

    train(args, dataloader, rank, model, optimizer, scheduler, saving_path)
    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', action='store', type=str)
    parser.add_argument('--name_of_dataset', action='store', type=str)
    parser.add_argument('--name_of_proj', action='store', type=str)
    parser.add_argument('--name_of_exp', action='store', type=str)

    parser.add_argument('--checkpoint', action='store', type=bool, default=False)
    parser.add_argument('--k_mer', action='store', type=int, default=4)
    parser.add_argument('--max_mask_count', action='store', type=int, default=80)
    parser.add_argument('--max_len', action='store', type=int, default=512)
    parser.add_argument('--batch_size', action='store', type=int, default=8)
    parser.add_argument('--lr', action='store', type=float, default=0.0005)
    parser.add_argument('--betas_a', action='store', type=float, default=0.9)
    parser.add_argument('--betas_b', action='store', type=float, default=0.98)
    parser.add_argument('--eps', action='store', type=float, default=1e-06)
    parser.add_argument('--weight_decay', action='store', type=float, default=1e-05)
    parser.add_argument('--loss_weight', action='store', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--name_of_run', type=str, required=False)
    parser.add_argument('--activate_wandb', default=False, action='store_true')
    parser.add_argument('--activate_lr_scheduler', default=False, action='store_true')
    parser.add_argument('--div_factor', default=10, required=False, type=float)

    args = vars(parser.parse_args())

    sys.stdout.write("\nTraining Parameters:\n")
    for key in args:
        sys.stdout.write(f'{key} \t -> {args[key]}\n')

    world_size = torch.cuda.device_count()
    saving_path = os.path.join("model_checkpoints", args['name_of_dataset'], args['name_of_exp'])

    os.makedirs(saving_path, exist_ok=True)
    mp.spawn(main, args=(world_size, args, saving_path), nprocs=world_size, join=True)
