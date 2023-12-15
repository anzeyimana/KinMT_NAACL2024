from __future__ import print_function, division

import gc
import math
import os
import os.path
import sys
import time
from shutil import copyfile
from typing import Union

import apex
import progressbar
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from arguments import py_trainer_args
from kinmt_data import KINMTDataCollection, KINMTDataset, read_parallel_data_files, \
    pick_first_collate_fn
from kinmt_en2kin import En2KinTransformer
from mygpt import MyGPTEncoder, MyGPT_from_pretrained
from modules import BaseConfig
from learning_rates import InverseSQRT_LRScheduler
from misc_functions import time_now
from fairseq.models.roberta import RobertaModel


def valid_loop(model: Union[DDP,En2KinTransformer], engl_roberta_model : Union[RobertaModel,None], kin_lm_model: Union[MyGPTEncoder,None],
               device, accumulation_steps, data_loader, args):
    world_size = dist.get_world_size()
    model.eval()
    model.zero_grad(set_to_none=True)

    NUM_LOSSES = (5 if args.kinmt_use_copy_loss else 4)
    loss_aggr = [torch.tensor(0.0, device=device) for _ in range(NUM_LOSSES)]
    nll_loss_aggr = [torch.tensor(0.0, device=device) for _ in range(NUM_LOSSES)]

    # Train
    count_items = 0

    num_data_items = torch.tensor(len(data_loader), device=device)
    dist.all_reduce(num_data_items, op=dist.ReduceOp.MIN)
    total_data_items = int(num_data_items.cpu().item())
    print(dist.get_rank(),f'Evaluating on {total_data_items} batches out of {len(data_loader)}')

    for batch_idx, batch_data_item in enumerate(data_loader):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            losses, nll_losses = model(engl_roberta_model, kin_lm_model, batch_data_item) #copy_batch_item_to_device(batch_data_item, device))
            mt_losses = [(loss/accumulation_steps) for loss in losses]
        for i in range(len(mt_losses)):
            loss_aggr[i] += (mt_losses[i].detach().clone().squeeze() * accumulation_steps)
        for i in range(len(nll_losses)):
            nll_loss_aggr[i] += (nll_losses[i].detach().clone().squeeze())
        count_items += 1
        if batch_idx == (total_data_items - 1):
            break

    loss_Z = count_items * world_size
    # Aggregate losses
    for i in range(len(loss_aggr)):
        dist.all_reduce(loss_aggr[i])

    for i in range(len(nll_loss_aggr)):
        dist.all_reduce(nll_loss_aggr[i])

    loss1 = nll_loss_aggr[0].item()/loss_Z
    loss2 = nll_loss_aggr[1].item()/loss_Z
    loss3 = nll_loss_aggr[2].item()/loss_Z
    loss4 = loss_aggr[3].item()/loss_Z

    tot_loss = loss1 + loss2 + loss3 + loss4

    return (tot_loss,
            loss1,
            loss2,
            loss3,
            loss4)

def train_loop(model: Union[DDP,En2KinTransformer], engl_roberta_model : Union[RobertaModel,None], kin_lm_model: Union[MyGPTEncoder,None],
               device, scaler: torch.cuda.amp.GradScaler, optimizer: apex.optimizers.FusedAdam, lr_scheduler: InverseSQRT_LRScheduler, data_loader,
               save_file_path, accumulation_steps, epoch, num_epochs, total_steps, best_valid_loss, bar, args):
    world_size = dist.get_world_size()
    model.train()
    model.zero_grad(set_to_none=True)

    NUM_LOSSES = (5 if args.kinmt_use_copy_loss else 4)
    loss_aggr = [torch.tensor(0.0, device=device) for _ in range(NUM_LOSSES)]
    nll_loss_aggr = [torch.tensor(0.0, device=device) for _ in range(NUM_LOSSES)]

    # Train
    start_steps = total_steps
    start_time = time.time()
    count_items = 0

    num_data_items = torch.tensor(len(data_loader), device=device)
    dist.all_reduce(num_data_items, op=dist.ReduceOp.MIN)
    total_data_items = int(num_data_items.cpu().item())
    print(dist.get_rank(),f'Training on {total_data_items} batches out of {len(data_loader)}')

    for batch_idx, batch_data_item in enumerate(data_loader):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            losses, nll_losses = model(engl_roberta_model, kin_lm_model, batch_data_item) #copy_batch_item_to_device(batch_data_item, device))
            total_loss = sum(losses)
            loss = total_loss / accumulation_steps
        scaler.scale(loss).backward()
        for i in range(len(losses)):
            loss_aggr[i] += (losses[i].detach().clone().squeeze())
        for i in range(len(nll_losses)):
            nll_loss_aggr[i] += (nll_losses[i].detach().clone().squeeze())
        total_steps += 1
        count_items += 1
        left_items = total_data_items - count_items
        if int(total_steps % (accumulation_steps//world_size)) == 0:
            lr_scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            current_time = time.time()
            #torch.cuda.empty_cache()
            if (dist.get_rank() == 0):
                print(time_now(),
                      'Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
                      'Warmup Iters: ', "{}".format(lr_scheduler.warmup_iter),
                      'OBJ:',
                      'STEM:', "{:.6f}".format(loss_aggr[0].item() / count_items),
                      'POS:', "{:.6f}".format(loss_aggr[1].item() / count_items),
                      'AFSET:', "{:.6f}".format(loss_aggr[2].item() / count_items),
                      'AFFIX:', "{:.6f}".format(loss_aggr[3].item() / count_items),
                      'NLL_OBJ:',
                      'STEM:', "{:.6f}".format(nll_loss_aggr[0].item() / count_items),
                      'POS:', "{:.6f}".format(nll_loss_aggr[1].item() / count_items),
                      'AFSET:', "{:.6f}".format(nll_loss_aggr[2].item() / count_items),
                      'LR: ', "{:.6f}/{}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
                      'Milli_Steps_Per_Second (MSS): ', "{:.3f}".format(1000.0 * ((total_steps - start_steps) / (accumulation_steps//world_size)) / (current_time - start_time)),
                      'Epochs:', '{}/{}'.format(epoch + 1, num_epochs), flush=True)
                bar.update(epoch)
                bar.fd.flush()
                sys.stdout.flush()
                sys.stderr.flush()
                if (lr_scheduler.num_iters % 5000) == 0:
                    if (math.isfinite((loss_aggr[0].item())) and
                            math.isfinite((loss_aggr[1].item())) and
                            math.isfinite((loss_aggr[2].item()))):
                        model.eval()
                        model.zero_grad(set_to_none=True)
                        with torch.no_grad():
                            torch.save({'model_state_dict': model.module.state_dict(),
                                        'scaler_state_dict': scaler.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                                        'epoch': (epoch + 1),
                                        'best_valid_loss': best_valid_loss,
                                        'num_epochs': num_epochs}, save_file_path+"_safe_checkpoint.pt")
                        print(time_now(), 'Safe model checkpointed!', flush=True)
                        model.train()
            if left_items < (accumulation_steps//world_size):
                break
        if batch_idx == (total_data_items - 1):
            break

    loss_Z = count_items * world_size
    # Aggregate losses
    for i in range(NUM_LOSSES):
        dist.all_reduce(loss_aggr[i])

    for i in range(NUM_LOSSES):
        dist.all_reduce(nll_loss_aggr[i])

    # Logging & Checkpointing
    if (dist.get_rank() == 0):
        print(time_now(),
              'After Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
              'LOSS:',
              'STEM:', "{:.6f}".format(loss_aggr[0].item()/loss_Z),
              'POS:', "{:.6f}".format(loss_aggr[1].item()/loss_Z),
              'AFSET:', "{:.6f}".format(loss_aggr[2].item()/loss_Z),
              'AFFIX:', "{:.6f}".format(loss_aggr[3].item()/loss_Z),
              'NLL_LOSS:',
              'STEM:', "{:.6f}".format(nll_loss_aggr[0].item()/loss_Z),
              'POS:', "{:.6f}".format(nll_loss_aggr[1].item()/loss_Z),
              'AFSET:', "{:.6f}".format(nll_loss_aggr[2].item()/loss_Z),
              'LR:', "{:.8f}/{:.5f}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
              'Warmup Iters:', "{}".format(lr_scheduler.warmup_iter),
              'Epochs:', '{}/{}'.format(epoch+1, num_epochs), flush=True)
        sys.stdout.flush()

        if os.path.exists(save_file_path):
            copyfile(save_file_path, save_file_path+"_prev_checkpoint.pt")
            print(time_now(), 'Prev model file checkpointed!', flush=True)

        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            torch.save({'model_state_dict': model.module.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'epoch': (epoch+1),
                        'best_valid_loss': best_valid_loss,
                        'num_epochs': num_epochs}, save_file_path)

    return total_steps

def train_fn(rank, args, cfg:BaseConfig, data_train, data_valid):
    device = torch.device('cuda:%d' % rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(rank)
    scaler = torch.cuda.amp.GradScaler()

    # For en2kin takes twice the memory vs kin2en: No need when not using Gradient Vaccine
    # args.kinmt_batch_max_tokens = args.kinmt_batch_max_tokens // 2
    # args.kinmt_accumulation_steps = args.kinmt_accumulation_steps * 2

    args.kinmt_num_train_epochs = args.kinmt_num_train_epochs // dist.get_world_size()

    if rank==0:
        print('Using device: ', device)

    home_path = args.home_path

    peak_lr = args.kinmt_peak_lr  # 0.001

    if (dist.get_rank() == 0):
        print('Model Arguments:', args)
        print(time_now(), 'Forming model ...', flush=True)

    if args.kinmt_use_gpt:
        my_gpt = MyGPT_from_pretrained(args, cfg, 'mygpt_final_2022-12-23_operated_full_base_2022-12-13.pt').to(device)
        my_gpt.float()
        my_gpt.eval()
        kin_lm_model = my_gpt.encoder
    else:
        kin_lm_model = None
    if args.kinmt_use_bert:
        engl_roberta_model = RobertaModel.from_pretrained('roberta.base', checkpoint_file='model.pt').to(device)
        engl_roberta_model.float()
        engl_roberta_model.eval()
    else:
        engl_roberta_model = None

    model = En2KinTransformer(args, cfg, engl_roberta_model, kin_lm_model, use_cross_pos_attn=args.use_cross_positional_attn_bias).to(device)
    model.float()
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.float()

    curr_save_file_path = home_path + f"data/{args.kinmt_model_name}.pt"

    if (dist.get_rank() == 0):
        print('---------------------------------- En2Kin Model Size ----------------------------------------')
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(time_now(), 'Total params:', total_params, 'Trainable params:', trainable_params, flush=True)
        print('Saving model in:', curr_save_file_path)
        print('---------------------------------------------------------------------------------------')

    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-08)

    if (dist.get_rank() == 0):
        print(time_now(), 'Reading parallel data ...', flush=True)

    # data_collection = KINMTDataCollection(cfg, english_ext='_en.txt',
    #                                       use_names_data=args.kinmt_use_names_data,
    #                                       use_foreign_terms=args.kinmt_use_foreign_terms,
    #                                       use_eval_data=args.kinmt_use_eval_data,
    #                                       kinmt_extra_train_data_key=args.kinmt_extra_train_data_key)
    #
    # train_dataset = KINMTDataset(data_collection.train_data + ((data_collection.lexical_data) * (args.kinmt_lexical_multiplier)),
    #                              english_ext='_en.txt',
    #                              max_tokens_per_batch=args.kinmt_batch_max_tokens,
    #                              randomized=True,
    #                              predict_affixes=True)
    #
    # train_data_loader = DataLoader(train_dataset, batch_size=1,
    #                                collate_fn=kinmt_en2kin_data_collate_fn,
    #                                drop_last=False, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)
    #
    # valid_dataset = KINMTDataset(data_collection.valid_data,
    #                              max_tokens_per_batch=args.kinmt_batch_max_tokens,
    #                              english_ext='_en.txt',
    #                              randomized=False,
    #                              predict_affixes=True)
    #
    # valid_data_loader = DataLoader(valid_dataset, batch_size=1,
    #                                collate_fn=kinmt_en2kin_data_collate_fn,
    #                                drop_last=False, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)

    train_dataset = KINMTDataset(data_train, rank,
                                 max_tokens_per_batch=args.kinmt_batch_max_tokens,
                                 english_ext='_en.txt',
                                 randomized=True,
                                 short_data_last=False,
                                 predict_affixes=True)

    valid_dataset = KINMTDataset(data_valid, rank,
                                 max_tokens_per_batch=args.kinmt_batch_max_tokens,
                                 english_ext='_en.txt',
                                 randomized=False,
                                 predict_affixes=True)

    train_data_loader = DataLoader(train_dataset, batch_size=1,
                                   collate_fn=pick_first_collate_fn,
                                   drop_last=False, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=False)

    valid_data_loader = DataLoader(valid_dataset, batch_size=1,
                                   collate_fn=pick_first_collate_fn,
                                   drop_last=False, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=False)

    if (dist.get_rank() == 0):
        print(time_now(), 'Done reading parallel data!', flush=True)

    iters_per_epoch = len(train_data_loader) // (args.kinmt_accumulation_steps // dist.get_world_size())

    lr_scheduler = InverseSQRT_LRScheduler(optimizer,
                                           start_lr=peak_lr,
                                           warmup_iter=args.kinmt_warmup_steps,
                                           num_iters=(iters_per_epoch * args.kinmt_num_train_epochs),
                                           last_iter=-1)
    best_valid_loss = 99999999.9

    if (not args.load_saved_model) and (dist.get_world_size() > 1):
        if (dist.get_rank() == 0):
            model.eval()
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                torch.save({'model_state_dict': model.module.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'epoch': 0,
                            'best_valid_loss': best_valid_loss,
                            'num_epochs': args.kinmt_num_train_epochs}, curr_save_file_path)
        dist.barrier()
        args.load_saved_model = True

    init_epoch = 0
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if args.load_saved_model:
        # Load saved state
        if (dist.get_rank() == 0):
            print(time_now(), 'Loading model state...', flush=True)
        kb_state_dict = torch.load(curr_save_file_path, map_location=map_location)
        model.module.load_state_dict(kb_state_dict['model_state_dict'])
        init_epoch = kb_state_dict['epoch']
        del kb_state_dict
        gc.collect()

        if (dist.get_rank() == 0):
            print(time_now(), 'Loading optimizer state...', flush=True)
        kb_state_dict = torch.load(curr_save_file_path, map_location=torch.device('cpu'))
        scaler.load_state_dict(kb_state_dict['scaler_state_dict'])
        optimizer.load_state_dict(kb_state_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(kb_state_dict['lr_scheduler_state_dict'])
        del kb_state_dict
        gc.collect()

    lr_scheduler.end_iter = (iters_per_epoch * args.kinmt_num_train_epochs)
    init_epoch = int(round(lr_scheduler.num_iters / iters_per_epoch))

    if (dist.get_rank() == 0):
        print('---------------------------------------------------------------------------------------')
        print('Model Arguments:', args)
        print('------------------ Train Config --------------------')
        print('epoch:', init_epoch)
        print('num_epochs:', args.kinmt_num_train_epochs)
        print('iters_per_epoch:', iters_per_epoch)
        print('Iters:', lr_scheduler.num_iters)
        print('End_Iter:', lr_scheduler.end_iter)
        print('Warmup_Iters:', lr_scheduler.warmup_iter)
        print('number_of_load_batches:', len(train_data_loader))
        print('accumulation_steps:', args.kinmt_accumulation_steps)
        print('batch_size:', args.kinmt_batch_max_tokens)
        print('effective_batch_size:', args.kinmt_batch_max_tokens * args.kinmt_accumulation_steps)
        print('peak_lr: {:.8f}'.format(peak_lr))
        print('-----------------------------------------------------')


    total_steps = int(lr_scheduler.num_iters * args.kinmt_accumulation_steps)

    init_epoch = (lr_scheduler.num_iters // iters_per_epoch)

    if (dist.get_rank() == 0):
        print(time_now(), 'Start training (total steps: {}) ....'.format(total_steps), flush=True)

    with progressbar.ProgressBar(initial_value=init_epoch,
                                 max_value=args.kinmt_num_train_epochs,
                                 redirect_stdout=True) as bar:
        if (dist.get_rank() == 0):
            bar.update(init_epoch)
            sys.stdout.flush()
        for epoch in range(init_epoch,args.kinmt_num_train_epochs):
            total_steps = train_loop(model, engl_roberta_model, kin_lm_model,
                                     device, scaler, optimizer, lr_scheduler, train_data_loader,
                                     curr_save_file_path, args.kinmt_accumulation_steps,
                                     epoch, args.kinmt_num_train_epochs, total_steps, best_valid_loss, bar, args)
            (tot_valid_loss,
             loss1,
             loss2,
             loss3,
             loss4) = valid_loop(model, engl_roberta_model, kin_lm_model,
                                 device, 1, valid_data_loader, args)
            if tot_valid_loss < best_valid_loss:
                best_valid_loss = tot_valid_loss
                model.eval()
                model.zero_grad(set_to_none=True)
                if dist.get_rank() == 0:
                    with torch.no_grad():
                        torch.save({'model_state_dict': model.module.state_dict(),
                                    'scaler_state_dict': scaler.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                                    'epoch': (epoch + 1),
                                    'best_valid_loss': best_valid_loss,
                                    'num_epochs': args.kinmt_num_train_epochs}, curr_save_file_path+"_best_valid_loss.pt")

            if dist.get_rank() == 0:
                print(time_now(),
                      'After Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
                      'VALID LOSS:',
                      'TOTAL:', "{:.6f}".format(tot_valid_loss),
                      'BEST:', "{:.6f}".format(best_valid_loss),
                      'STEM:', "{:.6f}".format(loss1),
                      'POS:', "{:.6f}".format(loss2),
                      'AFSET:', "{:.6f}".format(loss3),
                      'AFFIX:', "{:.6f}".format(loss4),
                      'LR:', "{:.8f}/{:.5f}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
                      'Warmup Iters:', "{}".format(lr_scheduler.warmup_iter),
                      'Epochs:', '{}/{}'.format(epoch + 1, args.kinmt_num_train_epochs), flush=True)
                print(time_now(), (epoch + 1), 'TRAINING EPOCHS COMPLETE!', flush=True)
                bar.update(epoch + 1)
                sys.stdout.flush()

def en2kin_trainer_main():
    args = py_trainer_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    if args.gpus == 0:
        args.world_size = 1

    cfg = BaseConfig()
    print('BaseConfig: \n\ttot_num_stems: {}\n'.format(cfg.tot_num_stems),
          '\ttot_num_affixes: {}\n'.format(cfg.tot_num_affixes),
          '\ttot_num_lm_morphs: {}\n'.format(cfg.tot_num_lm_morphs),
          '\ttot_num_pos_tags: {}\n'.format(cfg.tot_num_pos_tags), flush=True)


    data_collection = KINMTDataCollection(cfg, english_ext='_en.txt',
                                          use_names_data=args.kinmt_use_names_data,
                                          use_foreign_terms=args.kinmt_use_foreign_terms,
                                          use_eval_data=args.kinmt_use_eval_data,
                                          kinmt_extra_train_data_key=args.kinmt_extra_train_data_key)
    train_data = data_collection.train_data + ((data_collection.lexical_data) * (args.kinmt_lexical_multiplier))
    valid_data = data_collection.valid_data

    data_train = read_parallel_data_files("kinmt/parallel_data_2022/txt", train_data, 512, english_ext='_en.txt')
    data_valid = read_parallel_data_files("kinmt/parallel_data_2022/txt", valid_data, 512, english_ext='_en.txt')

    mp.spawn(train_fn, nprocs=args.world_size, args=(args, cfg, data_train, data_valid,))

if __name__ == '__main__':
    en2kin_trainer_main()
