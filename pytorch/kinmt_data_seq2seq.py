import os
import random
from datetime import datetime

import numpy as np
import torch
from modules import BaseConfig
from misc_functions import read_lines, time_now
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD_IDX = 1

def prepare_sentence_data(line, max_seq_len):
    tokens = [(int(tk),None) for tk in line.split('\t')]
    if len(tokens) > max_seq_len:
        tokens = [tokens[0]] + tokens[1:(max_seq_len-1)] + [tokens[-1]]
    return tokens

def read_parallel_data_files(data_dir, keywords):
    data = []
    for keyword in keywords:
        rw_lines = read_lines(data_dir + '/spm_parsed_' + keyword + '_rw.txt')
        en_lines = read_lines(data_dir + '/spm_parsed_' + keyword +'_en.txt')
        assert len(rw_lines)==len(en_lines), "Mismatch number of lines @ {} > rw: {} vs en: {}".format(keyword, len(rw_lines), len(en_lines))
        data.extend([(((rw.count('\t')+en.count('\t'))//2)+1 , rw , en) for rw,en in zip(rw_lines,en_lines)])
    return data

global_cfg = BaseConfig()

class KINMTDataCollection():

    def __init__(self, cfg: BaseConfig,
                 use_names_data = True,
                 use_foreign_terms = False,
                 use_eval_data = False,
                 kinmt_extra_train_data_key=None):
        global global_cfg
        global_cfg = cfg
        lexdata = ['habumuremyi_combined', 'kinyarwandanet', 'stage1', 'stage2', 'stage3', 'stage3_residuals']

        self.lexical_data = lexdata

        trdata = ['jw', 'gazette', 'numeric_examples']
        if use_names_data:
            trdata.append('names_and_numbers')
        if use_foreign_terms:
            trdata.append('foreign_terms')

        if kinmt_extra_train_data_key is not None:
            if len(kinmt_extra_train_data_key) > 0:
                trdata.append(kinmt_extra_train_data_key)

        if use_eval_data:
            trdata.extend(['flores200_dev', 'mafand_mt_dev', 'tico19_dev', 'flores200_test', 'mafand_mt_test', 'tico19_test'])
        self.train_data = trdata

        self.valid_data = ['flores200_dev', 'mafand_mt_dev', 'tico19_dev']

class KINMTDataset(Dataset):

    def __init__(self, data,
                 data_dir="kinmt/parallel_data_2022/txt",
                 max_tokens_per_batch=25000, num_shuffle_buckets=200,
                 randomized = False):
        # 1. Sort
        self.max_tokens_per_batch = max_tokens_per_batch
        self.randomized = randomized
        print(time_now(), 'Dataset reading ...')
        self.data = read_parallel_data_files(data_dir, data)
        print(time_now(), 'Dataset sorting ...')
        self.data.sort(key=lambda x: x[0], reverse=True)
        # 2. Demarcate buckets
        total_tokens = sum([x[0] for x in self.data])
        bucket_size = (total_tokens // num_shuffle_buckets) + 1
        start = 0
        self.buckets = []
        count = 0
        for i,x in enumerate(self.data):
            if (count+x[0]) > bucket_size:
                self.buckets.append((start,i))
                count = 0
                start = i
            count += x[0]
        # Expand the last bucket to the end
        end = self.buckets[-1]
        self.buckets[-1] = (end[0], len(self.data))
        self.batches = []
        if self.randomized:
            self.shuffle_buckets_and_mark_batches()
        else:
            self.form_uniexamplar_batches()

    def shuffle_buckets_and_mark_batches(self):
        print(time_now(), 'Dataset shuffling ...')
        seed_val = datetime.now().microsecond + (13467 * os.getpid())
        seed_val = int(seed_val) % ((2 ** 32) - 1)
        np.random.seed(seed_val)
        random.seed(seed_val)
        torch.random.manual_seed(seed_val)
        # Shuffle within buckets
        for (start,end) in self.buckets:
            copy = self.data[start:end]
            random.shuffle(copy)
            self.data[start:end] = copy
        # Form batches
        self.batches = []
        start = 0
        count = 0
        ex = 0
        for i,x in enumerate(self.data):
            if (count + x[0]) > self.max_tokens_per_batch:
                self.batches.append((self.data,start,i))
                count = 0
                ex = 0
                start = i
            count += x[0]
            ex += 1

        random.shuffle(self.batches)
        print(time_now(), 'Batching DONE: got {} batches; discarded {} examples ({} tokens)'.format(len(self.batches), ex, count))

    def form_uniexamplar_batches(self):
        print(time_now(), 'Dataset batching ...')
        seed_val = datetime.now().microsecond + (13467 * os.getpid())
        seed_val = int(seed_val) % ((2 ** 32) - 1)
        np.random.seed(seed_val)
        random.seed(seed_val)
        torch.random.manual_seed(seed_val)
        # Shuffle within buckets
        for (start,end) in self.buckets:
            copy = self.data[start:end]
            random.shuffle(copy)
            self.data[start:end] = copy
        # Form batches
        self.batches = []
        for i, x in enumerate(self.data):
            self.batches.append((self.data, i, i+1))
        random.shuffle(self.batches)
        print(time_now(), 'Batching DONE: got {} batches; discarded {} examples ({} tokens)'.format(len(self.batches), 0, 0))

    def __len__(self):
        return len(self.batches) + (8 if self.randomized else 0)

    def __getitem__(self, idx):
        batch_idx = idx % len(self.batches)
        if (batch_idx == (len(self.batches) - 1)) and self.randomized:
            self.shuffle_buckets_and_mark_batches()
        batch_idx = idx % len(self.batches)
        return self.batches[batch_idx]

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_input_key_padding_mask(input_lengths, ignore_last=False):
    input_masks = [torch.zeros(length, dtype=torch.bool) for length in input_lengths]
    if ignore_last:
        for i in range(len(input_masks)):
            if len(input_masks[i]) > 0:
                input_masks[i][-1] = True
    input_masks_padded = pad_sequence(input_masks, batch_first=True, padding_value=1)  # Shape: N x S
    return input_masks_padded

def kinmt_process_batch(batch_items):
    max_seq_len = 512
    
    (data, start, end) = batch_items[0]

    batch_kin_tokens = []
    batch_kin_tokens_sequence_lengths = []

    batch_eng_tokens = []
    batch_eng_tokens_sequence_lengths = []

    for (_,rw_line,en_line) in data[start:end]:
        rw_tokens = prepare_sentence_data(rw_line, max_seq_len)
        en_tokens = prepare_sentence_data(en_line, max_seq_len)

        batch_kin_tokens.extend([tid for (tid,tk) in rw_tokens])
        batch_kin_tokens_sequence_lengths.append(len(rw_tokens))

        batch_eng_tokens.extend([tid for (tid,tk) in en_tokens])
        batch_eng_tokens_sequence_lengths.append(len(en_tokens))

    kinya_input_ids = torch.tensor(batch_kin_tokens)
    kinya_sequence_lengths = batch_kin_tokens_sequence_lengths

    english_input_ids = torch.tensor(batch_eng_tokens)
    english_sequence_lengths = batch_eng_tokens_sequence_lengths


    data_item = (kinya_input_ids, kinya_sequence_lengths,
                 english_input_ids, english_sequence_lengths)
    return data_item

def kinmt_seq2seq_en2kin_data_collate_fn(batch_items):
    data_item = kinmt_process_batch(batch_items)
    (tgt_input_ids, tgt_sequence_lengths,
     src_input_ids, src_sequence_lengths) = data_item

    seq_len = max(tgt_sequence_lengths)
    src_key_padding_mask = generate_input_key_padding_mask(src_sequence_lengths, ignore_last=False)
    tgt_key_padding_mask = generate_input_key_padding_mask(tgt_sequence_lengths, ignore_last=True)
    decoder_mask = generate_square_subsequent_mask(seq_len)

    batch_data_item = (src_input_ids, src_sequence_lengths,
                       tgt_input_ids, tgt_sequence_lengths,
                       src_key_padding_mask, tgt_key_padding_mask, decoder_mask)
    return batch_data_item

def kinmt_seq2seq_kin2en_data_collate_fn(batch_items):
    data_item = kinmt_process_batch(batch_items)
    (src_input_ids, src_sequence_lengths,
     tgt_input_ids, tgt_sequence_lengths) = data_item

    seq_len = max(tgt_sequence_lengths)
    src_key_padding_mask = generate_input_key_padding_mask(src_sequence_lengths, ignore_last=False)
    tgt_key_padding_mask = generate_input_key_padding_mask(tgt_sequence_lengths, ignore_last=True)
    decoder_mask = generate_square_subsequent_mask(seq_len)

    batch_data_item = (src_input_ids, src_sequence_lengths,
                       tgt_input_ids, tgt_sequence_lengths,
                       src_key_padding_mask, tgt_key_padding_mask, decoder_mask)
    return batch_data_item
