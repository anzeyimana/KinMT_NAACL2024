from __future__ import print_function, division

import gc
import os
import os.path
import sys
import time
from shutil import copyfile
from typing import Union
import math
import apex
import progressbar
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from arguments import py_trainer_args
from kinmt_old_data import KINMTDataCollection, KINMTDataset, kinmt_kin2en_data_collate_fn
from kinmt_kin2en import Kin2EnTransformer
from mybert import MyBERTEncoder, MyBERT_from_pretrained, \
    MyBERT_large_from_pretrained
from modules import BaseConfig
from learning_rates import InverseSQRT_LRScheduler
from misc_functions import time_now
from fairseq.models.transformer_lm import TransformerLanguageModel

def copy_batch_item_to_device(batch_data_item, device):
    (english_input_ids, english_sequence_lengths,
     lm_morphs, pos_tags, stems, input_sequence_lengths,
     afx_padded, m_masks_padded,
     affixes_prob, tokens_lengths,
     copy_tokens_prob,
     src_key_padding_mask, tgt_key_padding_mask, decoder_mask) = batch_data_item

    return (device, english_input_ids.to(device, dtype=torch.long), english_sequence_lengths,
         lm_morphs.to(device, dtype=torch.long), pos_tags.to(device, dtype=torch.long), stems.to(device, dtype=torch.long), input_sequence_lengths,
         afx_padded.to(device, dtype=torch.long), m_masks_padded.to(device),
         affixes_prob.to(device) if (affixes_prob is not None) else None, tokens_lengths,
            (copy_tokens_prob.to(device) if (copy_tokens_prob is not None) else None),
            src_key_padding_mask.to(device), tgt_key_padding_mask.to(device), decoder_mask.to(device))

def eval_loop(model: Union[DDP,Kin2EnTransformer],
               mybert_encoder: Union[MyBERTEncoder,None],
               english_lm: Union[TransformerLanguageModel,None],
              data_loader,
              device):
    world_size = dist.get_world_size()
    model.zero_grad(set_to_none=True)
    model.eval()

    loss_aggr = torch.tensor(0.0, device=device)
    nll_loss_aggr = torch.tensor(0.0, device=device)

    count_items = 0

    num_data_items = torch.tensor(len(data_loader), device=device)
    dist.all_reduce(num_data_items, op=dist.ReduceOp.MIN)
    total_data_items = int(num_data_items.item())
    print(dist.get_rank(),f'Evaluating on {total_data_items} batches out of {len(data_loader)}')

    with torch.no_grad():
        for batch_idx, batch_data_item in enumerate(data_loader):
            # with torch.cuda.amp.autocast():
            losses, nll_losses = model(mybert_encoder, english_lm, copy_batch_item_to_device(batch_data_item, device))
            loss_aggr += losses[0].detach().clone().squeeze()
            nll_loss_aggr += nll_losses[0].detach().clone().squeeze()
            count_items += 1
            if batch_idx == (total_data_items - 1):
                break

    loss_Z = count_items * world_size
    # Aggregate losses
    dist.all_reduce(loss_aggr)

    dist.all_reduce(nll_loss_aggr)

    nll_loss = nll_loss_aggr.item()/loss_Z
    loss = loss_aggr.item()/loss_Z

    return nll_loss, loss

def train_loop(model: Union[DDP,Kin2EnTransformer],
               mybert_encoder: Union[MyBERTEncoder,None],
               english_lm: Union[TransformerLanguageModel,None],
               device, scaler: torch.cuda.amp.GradScaler, optimizer: apex.optimizers.FusedAdam,
               lr_scheduler: InverseSQRT_LRScheduler, data_loader,
               save_file_path, accumulation_steps, epoch, num_epochs, total_steps, best_valid_loss, bar):
    min_scale = 256
    world_size = dist.get_world_size()
    model.train()
    model.zero_grad(set_to_none=True)

    loss_aggr = torch.tensor(0.0, device=device)
    nll_loss_aggr = torch.tensor(0.0, device=device)

    # Train
    started = False
    start_steps = total_steps
    start_time = time.time()
    real_total_steps = torch.tensor(1.0*total_steps, device=device)
    count_items = torch.tensor(0.0, device=device)

    num_data_items = torch.tensor(len(data_loader), device=device)
    dist.all_reduce(num_data_items, op=dist.ReduceOp.MIN)
    total_data_items = int(num_data_items.item())
    print(dist.get_rank(),f'Training on {total_data_items} batches out of {len(data_loader)}')

    batch_loss_aggr = torch.tensor(0.0, device=device)
    batch_nll_loss_aggr = torch.tensor(0.0, device=device)

    for batch_idx, batch_data_item in enumerate(data_loader):
        # with torch.cuda.amp.autocast():
        losses, nll_losses = model(mybert_encoder, english_lm, copy_batch_item_to_device(batch_data_item, device))
        loss = losses[0]/accumulation_steps
        # scaler.scale(loss).backward()
        loss.backward()

        loss_val = losses[0].item()
        nll_loss_val = nll_losses[0].item()
        batch_loss_aggr += loss_val
        batch_nll_loss_aggr += nll_loss_val

        if math.isfinite(loss_val) and math.isfinite(nll_loss_val):
            loss_aggr += loss_val
            nll_loss_aggr += nll_loss_val
            count_items += 1.0
            real_total_steps += 1.0

        total_steps += 1
        left_items = total_data_items - (batch_idx+1)

        if int(total_steps % (accumulation_steps // world_size)) == 0:
            dist.all_reduce(batch_loss_aggr)
            dist.all_reduce(batch_nll_loss_aggr)
            batch_loss = batch_loss_aggr.item()
            batch_nll_loss = batch_nll_loss_aggr.item()
            # Skip NaN losses
            if math.isfinite(batch_loss) and math.isfinite(batch_nll_loss):
                lr_scheduler.step()
                optimizer.step()
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
                # scaler.step(optimizer)
                # scaler.update()
                # if scaler._scale < min_scale:
                #     scaler._scale = torch.tensor(min_scale).to(scaler._scale)
            else:
                if (dist.get_rank() == 0):
                    print(time_now(),
                          'Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
                          f"batch_loss: {batch_loss}, batch_nll_loss: {batch_nll_loss}",flush=True)
            optimizer.zero_grad()
            current_time = time.time()
            # torch.cuda.empty_cache()
            if (dist.get_rank() == 0) and math.isfinite(batch_loss) and math.isfinite(batch_nll_loss):
                if (int(lr_scheduler.num_iters) % 10) == 0:
                    print(time_now(),
                          'Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
                          'Warmup Iters: ', "{}".format(lr_scheduler.warmup_iter),
                          'OBJ:',
                          'TOKEN:', "{:.6f}".format(loss_aggr.item() / count_items.item()),
                          'NLL_OBJ:',
                          'TOKEN:', "{:.6f}".format(nll_loss_aggr.item() / count_items.item()),
                          'LR: ', "{:.6f}/{}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
                          'Milli_Steps_Per_Second (MSS): ', "{:.3f}".format(1000.0 * ((real_total_steps.item() - start_steps) / (accumulation_steps//world_size)) / (current_time - start_time)),
                          'Epochs:', '{}/{}'.format(epoch + 1, num_epochs), flush=True)
                    bar.update(epoch)
                    bar.fd.flush()
                    sys.stdout.flush()
                    sys.stderr.flush()
                if (lr_scheduler.num_iters % 1000) == 0:
                    if math.isfinite((loss_aggr.item())):
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
            batch_loss_aggr.zero_()
            batch_nll_loss_aggr.zero_()
            if left_items < (accumulation_steps // world_size):
                break
            if not started:
                started = True
                start_steps = total_steps
                start_time = time.time()
        if batch_idx == (total_data_items - 1):
            break

    dist.all_reduce(count_items)
    loss_Z = count_items.item()
    # Aggregate losses
    dist.all_reduce(loss_aggr)
    dist.all_reduce(nll_loss_aggr)

    # Logging & Checkpointing
    if dist.get_rank() == 0:
        print(time_now(),
              'After Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
              'LOSS:',
              'TOKEN:', "{:.6f}".format(loss_aggr.item()/loss_Z),
              'NLL_LOSS:',
              'STEM:', "{:.6f}".format(nll_loss_aggr.item()/loss_Z),
              'LR:', "{:.8f}/{:.5f}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
              'Warmup Iters:', "{}".format(lr_scheduler.warmup_iter),
              'Epochs:', '{}/{}'.format(epoch+1, num_epochs), flush=True)
        sys.stdout.flush()

        if os.path.exists(save_file_path):
            copyfile(save_file_path, save_file_path + "_prev_checkpoint.pt")
            print(time_now(), 'Prev model file checkpointed!', flush=True)

        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            torch.save({'model_state_dict': model.module.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'epoch': (epoch + 1),
                        'best_valid_loss': best_valid_loss,
                        'num_epochs': num_epochs}, save_file_path)

    return total_steps

def train_fn(rank, args, cfg: BaseConfig):
    print(time_now(), 'Called train_fn()', flush=True)
    device = torch.device('cuda:%d' % rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(rank)

    dist.barrier()
    print('Using device: ', device, "from", dist.get_world_size(), 'processes', flush=True)

    scaler = torch.cuda.amp.GradScaler(growth_interval=100)
    args.kinmt_num_train_epochs = args.kinmt_num_train_epochs // dist.get_world_size()

    adam_eps = 1e-07

    home_path = args.home_path

    peak_lr = args.kinmt_peak_lr  # 0.001

    if (dist.get_rank() == 0):
        print('Model Arguments:', args)
        print(time_now(), 'Forming model ...', flush=True)

    if args.kinmt_use_bert:
        if args.kinmt_bert_large:
            mybert = MyBERT_large_from_pretrained(device, args, cfg,'mybert_large_2023-06-25.pt_160K.pt').to(device)
        else:
            mybert = MyBERT_from_pretrained(device, args, cfg, 'mybert_base_2023-06-06.pt_160K.pt').to(device)
        mybert.float()
        mybert.eval()
        mybert_encoder = mybert.encoder
    else:
        mybert_encoder = None

    if args.kinmt_use_gpt:
        english_transformer_model = TransformerLanguageModel.from_pretrained('wmt19.en', 'model.pt',
                                                                             tokenizer='moses', bpe='fastbpe').to(device)
        english_transformer_model.float()
        english_transformer_model.eval()
        english_lm = english_transformer_model.models[0]
    else:
        english_lm = None

    model = Kin2EnTransformer(args, cfg, mybert_encoder, english_lm, use_cross_pos_attn=args.use_cross_positional_attn_bias).to(device)
    model.float()
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.float()

    curr_save_file_path = home_path + f"data/{args.kinmt_model_name}.pt"

    if (dist.get_rank() == 0):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('---------------------------------- Kin2En Model Size ----------------------------------------')
        print(time_now(), 'Total params:', total_params, 'Trainable params:', trainable_params, flush=True)
        print('Saving model in:', curr_save_file_path)
        print('---------------------------------------------------------------------------------------')

    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=adam_eps)

    if (dist.get_rank() == 0):
        print(time_now(), 'Reading parallel data ...', flush=True)

    data_collection = KINMTDataCollection(cfg, english_ext='_lm_en.txt',
                                          use_names_data=args.kinmt_use_names_data,
                                          use_foreign_terms=args.kinmt_use_foreign_terms,
                                          use_eval_data=args.kinmt_use_eval_data,
                                          kinmt_extra_train_data_key=args.kinmt_extra_train_data_key)

    train_dataset = KINMTDataset(data_collection.train_data + ((data_collection.lexical_data) * (args.kinmt_lexical_multiplier)),
                                 max_tokens_per_batch=args.kinmt_batch_max_tokens,
                                 english_ext='_lm_en.txt',
                                 randomized=True)
    valid_dataset = KINMTDataset(data_collection.valid_data,
                                 max_tokens_per_batch=args.kinmt_batch_max_tokens,
                                 english_ext='_lm_en.txt',
                                 randomized=False)
    # train_dataset = KINMTDataset(data_train,
    #                              max_tokens_per_batch=args.kinmt_batch_max_tokens,
    #                              english_ext='_lm_en.txt',
    #                              randomized=True)
    #
    # valid_dataset = KINMTDataset(data_valid,
    #                              max_tokens_per_batch=args.kinmt_batch_max_tokens,
    #                              english_ext='_lm_en.txt',
    #                              randomized=False)

    train_data_loader = DataLoader(train_dataset, batch_size=1,
                                   collate_fn=kinmt_kin2en_data_collate_fn,
                                   drop_last=False, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)


    valid_data_loader = DataLoader(valid_dataset, batch_size=1,
                                   collate_fn=kinmt_kin2en_data_collate_fn,
                                   drop_last=False, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)

    if (dist.get_rank() == 0):
        print(time_now(), 'Done reading parallel data!', flush=True)

    iters_per_epoch = len(train_data_loader) // (args.kinmt_accumulation_steps // dist.get_world_size())

    lr_scheduler = InverseSQRT_LRScheduler(optimizer,
                                           start_lr=peak_lr,
                                           warmup_iter=args.kinmt_warmup_steps,
                                           num_iters=(iters_per_epoch * args.kinmt_num_train_epochs),
                                           last_iter=-1)
    best_valid_loss = 9999999.99

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

    epoch = 0
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if args.load_saved_model:
        # Load saved state
        if (dist.get_rank() == 0):
            print(time_now(), 'Loading model state...', flush=True)
        kb_state_dict = torch.load(curr_save_file_path, map_location=map_location)
        kb_state_dict['optimizer_state_dict']['param_groups'][0]['eps'] = adam_eps
        model.module.load_state_dict(kb_state_dict['model_state_dict'])
        epoch = kb_state_dict['epoch']
        best_valid_loss = kb_state_dict['best_valid_loss']
        del kb_state_dict
        gc.collect()

        if (dist.get_rank() == 0):
            print(time_now(), 'Loading optimizer state...', flush=True)
        kb_state_dict = torch.load(curr_save_file_path, map_location=torch.device('cpu'))
        kb_state_dict['optimizer_state_dict']['param_groups'][0]['eps'] = adam_eps
        scaler.load_state_dict(kb_state_dict['scaler_state_dict'])
        optimizer.load_state_dict(kb_state_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(kb_state_dict['lr_scheduler_state_dict'])
        del kb_state_dict
        gc.collect()

    lr_scheduler.end_iter = (iters_per_epoch * args.kinmt_num_train_epochs)
    epoch = int(math.floor(lr_scheduler.num_iters / iters_per_epoch))

    lr_scheduler.start_lr = peak_lr

    if (dist.get_rank() == 0):
        print('---------------------------------------------------------------------------------------')
        print('Model Arguments:', args)
        print('------------------ Train Config --------------------')
        print('epoch:', epoch)
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

    if (dist.get_rank() == 0):
        print(time_now(), 'Start training (total steps: {}) ....'.format(total_steps), flush=True)
    with progressbar.ProgressBar(initial_value=epoch,
                                 max_value=args.kinmt_num_train_epochs,
                                 redirect_stdout=True) as bar:
        if (dist.get_rank() == 0):
            bar.update(epoch)
            sys.stdout.flush()
        for ep in range(epoch, args.kinmt_num_train_epochs):
            total_steps = train_loop(model, mybert_encoder, english_lm,
                                     device, scaler, optimizer, lr_scheduler, train_data_loader,
                                     curr_save_file_path, args.kinmt_accumulation_steps,
                                     ep, args.kinmt_num_train_epochs, total_steps, best_valid_loss, bar)

            (nll_loss, ls_loss) = eval_loop(model, mybert_encoder, english_lm, valid_data_loader, device)
            if nll_loss < best_valid_loss:
                best_valid_loss = nll_loss
                model.eval()
                model.zero_grad(set_to_none=True)
                if dist.get_rank() == 0:
                    with torch.no_grad():
                        torch.save({'model_state_dict': model.module.state_dict(),
                                    'scaler_state_dict': scaler.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                                    'epoch': (ep + 1),
                                    'best_valid_loss': best_valid_loss,
                                    'num_epochs': args.kinmt_num_train_epochs}, curr_save_file_path+"_best_valid_loss.pt")

            if dist.get_rank() == 0:
                print(time_now(),
                      'After Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
                      'VALID LOSS:',
                      'NLL TOKEN:', "{:.6f}".format(nll_loss),
                      'BEST:', "{:.6f}".format(best_valid_loss),
                      'LS TOKEN:', "{:.6f}".format(ls_loss),
                      'LR:', "{:.8f}/{:.5f}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
                      'Warmup Iters:', "{}".format(lr_scheduler.warmup_iter),
                      'Epochs:', '{}/{}'.format(ep + 1, args.kinmt_num_train_epochs), flush=True)
                print(time_now(), (ep + 1), 'TRAINING EPOCHS COMPLETE!', flush=True)
                bar.update(ep + 1)
                sys.stdout.flush()

def kin2en_trainer_main():
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

    mp.spawn(train_fn, nprocs=args.world_size, args=(args, cfg,))


if __name__ == '__main__':
    kin2en_trainer_main()
