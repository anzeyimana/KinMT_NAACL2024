import math
import multiprocessing as mp
from argparse import ArgumentParser
from itertools import accumulate
import os

import progressbar
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from morphoy import ParsedMorphoSentence
from arguments import py_trainer_args
from kinmt_data import prepare_kinya_sentence_data, generate_input_key_padding_mask
from kinmt_decode_kin2en import kin2en_init_models, kin2en_translate, \
    reset_and_print_kin2en_latency_info
from misc_functions import read_lines
from misc_functions import time_now


def process_input_item(input_sentence, device, tot_num_affixes = 403):
    predict_affixes = False
    parsed_sentence = ParsedMorphoSentence(input_sentence, delimeter=';')
    (batch_lm_morphs,
     batch_pos_tags,
     batch_affixes,
     tokens_lengths,
     batch_stems) = prepare_kinya_sentence_data(parsed_sentence, 512)

    input_sequence_lengths = [len(batch_stems)]

    src_key_padding_mask = generate_input_key_padding_mask(input_sequence_lengths, ignore_last=False).to(device)
    lm_morphs = torch.tensor(batch_lm_morphs).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    affixes = torch.tensor(batch_affixes)  # .to(device)
    stems = torch.tensor([sid for sid in batch_stems]).to(device)

    if predict_affixes:
        pred_affixes_list = [batch_affixes[x - y: x] for x, y in zip(accumulate(tokens_lengths), tokens_lengths)]
        afx_prob = torch.zeros(len(pred_affixes_list), tot_num_affixes)
        for i, lst in enumerate(pred_affixes_list):
            if (len(lst) > 0):
                afx_prob[i, lst] = 1.0
        affixes_prob = afx_prob.to(device, dtype=torch.float)
    else:
        affixes_prob = None

    afx = affixes.split(tokens_lengths)
    # [[2,4,5], [6,7]]
    afx_padded = pad_sequence(afx, batch_first=False)
    afx_padded = afx_padded.to(device)
    # afx_padded: (M,L), M: max morphological length
    m_masks_padded = None
    if afx_padded.nelement() > 0:
        m_masks = [torch.zeros((x + 4), dtype=torch.bool, device=stems.device) for x in tokens_lengths]
        m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1)  # Shape: (L, 4+M)

    seed_data_item = (lm_morphs, pos_tags, stems, input_sequence_lengths,
                      afx_padded, m_masks_padded,
                      affixes_prob, tokens_lengths,
                      src_key_padding_mask)
    return seed_data_item
class BTDataset(Dataset):
    def __init__(self, lines, device):
        self.inputs = lines
        self.device = device
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return process_input_item(self.inputs[idx], self.device)

def pick_first_collate_fn(items):
    return items[0]

def translate_corpus(rank, input_filename:str, output_filename:str, start_at=1, end_at=1000, parsed=True):
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(rank)

    args = py_trainer_args(list_args=[], silent=True)

    args.load_saved_model = True
    args.kinmt_use_bert = True
    args.kinmt_use_gpt = False
    args.use_cross_positional_attn_bias = True
    args.kinmt_use_copy_loss = False
    args.kinmt_bert_large = True
    args.kinmt_model_name = "kin2en_base_large_bert_trial_backtrans_eval_xpos_2023-07-22"

    kin2en_model_setup = kin2en_init_models(args, rank=rank, use_morpho_lib=(not parsed))

    print('Model: ready!')

    copy_threshold = 0.5
    intersect_threshold = 0.5
    beam_size = 4
    max_eval_sequences = 5

    (ffi, lib, cfg, device, kin2en_model, mybert_encoder, english_lm, english_decoder) = kin2en_model_setup

    print(time_now(), 'Reading input data ...', flush=True)
    existing_lines = 0
    if os.path.isfile(output_filename):
        output_lines = read_lines(output_filename)
        existing_lines = len(output_lines)
        del output_lines
    print(time_now(), f'Input data starts at {existing_lines}', flush=True)
    input_lines = read_lines(input_filename)
    input_lines = input_lines[existing_lines:]

    eval_dataset = BTDataset(input_lines, device)
    eval_data_loader = DataLoader(eval_dataset, batch_size=1,
                                  drop_last=False, shuffle=False, num_workers=1,
                                  collate_fn=pick_first_collate_fn,
                                  persistent_workers=False,
                                  pin_memory=False)
    print(time_now(), 'Data loader ready!', flush=True)

    with open(output_filename, 'a', encoding='utf-8') as out_file:
        with progressbar.ProgressBar(max_value=len(eval_data_loader), redirect_stdout=True) as bar:
            for item_idx,input_data_item in enumerate(eval_data_loader):
                if ((item_idx % 100) == 0):
                    reset_and_print_kin2en_latency_info()
                    bar.update(item_idx)
                input_length = input_data_item[2].shape[0] #len(kinya_sentence.split(';'))
                max_text_length = int(math.ceil(input_length * 2.5))
                (complete_translations,
                 pending_translations) = kin2en_translate(kin2en_model_setup,
                                                          input_data_item,
                                                          beam_size, max_eval_sequences, max_text_length,
                                                          ready_sentence_item=True,
                                                          copy_threshold=copy_threshold,
                                                          intersect_threshold=intersect_threshold)
                kinmt_translation = input_lines[item_idx]
                if len(complete_translations) > 0:
                    kinmt_translation = complete_translations[0][1]
                elif len(pending_translations) > 0:
                    kinmt_translation = pending_translations[0][1]

                out_file.write(f'{kinmt_translation}\n')
                out_file.flush()

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        pass
    parser = ArgumentParser(description="Kin2En BackTranslator")
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--n", type=int, default=1)
    proc_args = parser.parse_args()

    total_lines = 16_373_718
    n = proc_args.n
    idx = n - 1
    rank = idx % (proc_args.gpus)
    batch = int(math.ceil(total_lines/proc_args.procs))
    start_at = (idx * batch) + 1
    end_at = min((idx + 1) * batch, total_lines)

    translate_corpus(rank,
                     f'kinmt/parallel_data_2022/txt/parsed_morpho_corpus_sentences_clean_2023-07-27_{n}_{start_at}_{end_at}_rw.txt',
                     f'kinmt/parallel_data_2022/txt/morpho_corpus_sentences_clean_2023-07-27_{n}_{start_at}_{end_at}_en.txt',
                     start_at=start_at,
                     end_at=end_at,
                     parsed = True)
