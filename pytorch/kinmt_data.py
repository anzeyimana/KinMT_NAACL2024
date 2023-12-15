import gc
import random
from datetime import datetime
from itertools import accumulate
import os
import sys

import numpy as np
import torch
from morphoy import ParsedMorphoSentence, BOS_ID, EOS_ID
from modules import BaseConfig
from misc_functions import read_lines, time_now
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import progressbar

EN_PAD_IDX = 1
KIN_PAD_IDX = 0

class EnglishToken:
    def __init__(self, parsed_token):
        self.uses_bpe = False
        self.surface_form = '_'
        self.raw_surface_form = '_'
        if '|' in parsed_token:
            idx = parsed_token.index('|')
            tks = parsed_token[:idx].split(',')
            sfc = parsed_token[(idx+1):]
            if len(sfc)>0:
                self.surface_form = sfc
                self.raw_surface_form = sfc
                self.uses_bpe = True
        else:
            tks = parsed_token.split(',')
        self.token_ids = [int(v) if (len(v)>0) else 2 for v in tks]

def prepare_kinya_sentence_data(sentence: ParsedMorphoSentence, max_seq_len):
    lm_morphs = []
    pos_tags = []
    stems = []
    affixes = []
    tokens_lengths = []

    for token in sentence.tokens:
        lm_morphs.append(token.lm_morph_id)
        pos_tags.append(token.pos_tag_id)
        # if token.uses_bpe:
        #     stems.append((token.stem_id, token.surface_form))
        # else:
        stems.append(token.stem_id)
        affixes.extend(token.affixes)
        tokens_lengths.append(len(token.affixes))
        for tid in token.extra_tokens_ids:
            lm_morphs.append(token.lm_morph_id)
            pos_tags.append(token.pos_tag_id)
            stems.append(tid)
            affixes.extend(token.affixes)
            tokens_lengths.append(len(token.affixes))

    lm_morphs = [BOS_ID] + lm_morphs + [EOS_ID]
    pos_tags = [BOS_ID] + pos_tags + [EOS_ID]
    stems = [BOS_ID] + stems + [EOS_ID]
    tokens_lengths = [0] + tokens_lengths + [0]

    if len(stems) > max_seq_len:
        lm_morphs = [BOS_ID] + lm_morphs[1:(max_seq_len-1)] + [EOS_ID]
        pos_tags = [BOS_ID] + pos_tags[1:(max_seq_len-1)] + [EOS_ID]
        stems = [BOS_ID] + stems[1:(max_seq_len-1)] + [EOS_ID]
        tokens_lengths = [0] + tokens_lengths[1:(max_seq_len-1)] + [0]

    return (lm_morphs,
            pos_tags,
            affixes,
            tokens_lengths,
            stems)

def prepare_english_sentence_data(line, max_seq_len):
    tokens = [int(tk) for tk in line.split('\t')]
    if len(tokens) > max_seq_len:
        tokens = [tokens[0]] + tokens[1:(max_seq_len-1)] + [tokens[-1]]
    return tokens

def read_parallel_data_files(data_dir, keywords, max_seq_len, english_ext='_en.txt'):
    metadata = []
    global_lm_morphs = []
    global_pos_tags = []
    global_affixes = []
    global_tokens_lengths = []
    global_stems = []
    global_en_tokens = []
    start_rw = 0
    start_en = 0
    afx_start = 0
    for keyword in keywords:
        print(time_now(), f'Processing {keyword} ...', flush=True)
        rw_lines = read_lines(data_dir + '/parsed_' + keyword + '_rw.txt')
        en_lines = read_lines(data_dir + '/parsed_' + keyword + english_ext)
        assert len(rw_lines)==len(en_lines), "Mismatch number of lines @ {} > rw: {} vs en: {}".format(keyword, len(rw_lines), len(en_lines))
        print(time_now(), f'Appending {keyword} ...', flush=True)
        itr = 0
        with progressbar.ProgressBar(initial_value=0,
                                     max_value=len(en_lines),
                                     redirect_stdout=True) as bar:
            for rw_line, en_line in zip(rw_lines,en_lines):
                if  (itr % 100_000) == 0:
                    bar.update(itr)
                    bar.fd.flush()
                    sys.stdout.flush()
                    sys.stderr.flush()
                itr += 1
                (seq_lm_morphs,
                 seq_pos_tags,
                 seq_affixes,
                 seq_tokens_lengths,
                 seq_stems) = prepare_kinya_sentence_data(ParsedMorphoSentence(rw_line, delimeter=';'), max_seq_len)
                seq_en_tokens = prepare_english_sentence_data(en_line, max_seq_len)
                rw_len = len(seq_stems)
                en_len = len(seq_en_tokens)
                if (rw_len > 2) and (en_len > 2):
                    global_lm_morphs.extend(seq_lm_morphs)
                    global_pos_tags.extend(seq_pos_tags)
                    global_affixes.extend(seq_affixes)
                    global_tokens_lengths.extend(seq_tokens_lengths)
                    global_stems.extend(seq_stems)
                    global_en_tokens.extend(seq_en_tokens)
                    avg_len = int(((rw_len + en_len) // 2) + 1)
                    end_rw = start_rw + rw_len
                    end_en = start_en + en_len
                    afx_end = afx_start + len(seq_affixes)
                    metadata.append([avg_len, start_rw, end_rw, afx_start, afx_end, start_en, end_en])
                    start_rw = end_rw
                    start_en = end_en
                    afx_start = afx_end
    # Create tensors
    global_lm_morphs = torch.tensor(global_lm_morphs, dtype=torch.int32)
    gc.collect()
    global_pos_tags = torch.tensor(global_pos_tags, dtype=torch.int32)
    gc.collect()
    global_affixes = torch.tensor(global_affixes, dtype=torch.int32)
    gc.collect()
    global_tokens_lengths = torch.tensor(global_tokens_lengths, dtype=torch.int32)
    gc.collect()
    global_stems = torch.tensor(global_stems, dtype=torch.int32)
    gc.collect()
    global_en_tokens = torch.tensor(global_en_tokens, dtype=torch.int32)
    gc.collect()
    # Put tensors into shared memory
    global_lm_morphs.share_memory_()
    global_pos_tags.share_memory_()
    global_affixes.share_memory_()
    global_tokens_lengths.share_memory_()
    global_stems.share_memory_()
    global_en_tokens.share_memory_()
    # Put tensors together
    global_tensors = (global_lm_morphs, global_pos_tags, global_affixes, global_tokens_lengths, global_stems, global_en_tokens)
    return global_tensors, metadata

global_cfg = BaseConfig()

def en2kin_copy_batch_item_to_device(batch_data_item, device):
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

def kin2en_copy_batch_item_to_device(batch_data_item, device):
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

class KINMTDataCollection():

    def __init__(self, cfg: BaseConfig,
                 data_dir="kinmt/parallel_data_2022/txt",
                 english_ext='_en.txt',
                 max_seq_len=512,
                 use_names_data = True,
                 use_foreign_terms = False,
                 use_eval_data = False,
                 kinmt_extra_train_data_key=None):
        global global_cfg
        global_cfg = cfg
        lexdata = ['habumuremyi_combined', 'kinyarwandanet', 'stage1', 'stage2', 'stage3', 'stage3_residuals']

        self.lexical_data = lexdata#read_parallel_data_files(data_dir, lexdata, max_seq_len, english_ext=english_ext)
        # Remove alphabetical order
        #random.shuffle(self.lexical_data)

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
        self.train_data = trdata#read_parallel_data_files(data_dir, trdata, max_seq_len, english_ext=english_ext)
        #random.shuffle(self.train_data)

        self.valid_data = ['flores200_dev', 'mafand_mt_dev', 'tico19_dev']#read_parallel_data_files(data_dir, ['flores200_dev', 'mafand_mt_dev', 'tico19_dev'], max_seq_len, english_ext=english_ext)

        # self.flores200_test_data = read_parallel_data_files(data_dir, ['flores200_test'], max_seq_len, english_ext=english_ext)
        # self.tico19_test_data = read_parallel_data_files(data_dir, ['tico19_test'], max_seq_len, english_ext=english_ext)
        # self.mafand_mt_test_data = read_parallel_data_files(data_dir, ['mafand_mt_test'], max_seq_len, english_ext=english_ext)

class KINMTDataset(Dataset):

    def __init__(self, tensors_and_metadata, rank,
                 data_dir="kinmt/parallel_data_2022/txt",
                 max_seq_len=512,
                 english_ext='_en.txt',
                 max_tokens_per_batch=25000, num_shuffle_buckets=200,
                 randomized = False,
                 short_data_last=True,
                 predict_affixes=False):
        # 1. Sort
        self.rank = rank
        self.device = None
        self.predict_affixes=predict_affixes
        self.max_tokens_per_batch = max_tokens_per_batch
        self.randomized = randomized
        print(time_now(), 'Dataset reading ...')
        self.data_tensors, meta = tensors_and_metadata #read_parallel_data_files(data_dir, data, max_seq_len, english_ext=english_ext)
        gc.collect()
        print(time_now(), 'Dataset sorting ...')
        meta.sort(key=lambda x: x[0], reverse=short_data_last)
        total_tokens = sum([x[0] for x in meta])

        self.metadata = torch.tensor(meta, dtype=torch.int32)
        self.metadata.share_memory_()
        # 2. Demarcate buckets
        bucket_size = (total_tokens // num_shuffle_buckets) + 1
        start = 0
        self.buckets = []
        count = 0
        for i,x in enumerate(meta):
            if (count+x[0]) > bucket_size:
                self.buckets.append((start,i))
                count = 0
                start = i
            count += x[0]
        # Expand the last bucket to the end
        end = self.buckets[-1]
        self.buckets[-1] = (end[0], len(meta))
        self.batches = []
        if self.randomized:
            self.shuffle_buckets_and_mark_batches()
        else:
            self.form_uniexamplar_batches()
        del meta
        gc.collect()

    def shuffle_buckets_and_mark_batches(self):
        print(time_now(), 'Dataset shuffling ...')
        seed_val = datetime.now().microsecond + (13467 * os.getpid())
        seed_val = int(seed_val) % ((2 ** 32) - 1)
        np.random.seed(seed_val)
        random.seed(seed_val)
        torch.random.manual_seed(seed_val)
        # Shuffle within buckets
        for (start,end) in self.buckets:
            copy = self.metadata[start:end].tolist()
            random.shuffle(copy)
            self.metadata[start:end] = torch.tensor(copy, dtype=torch.int32)
        # Form batches
        self.batches = []
        start = 0
        count = 0
        ex = 0
        # for i,x in enumerate(self.metadata):
        for i in range(self.metadata.shape[0]):
            avg_len = int(self.metadata[i,0].item())
            if (count + avg_len) > self.max_tokens_per_batch:
                self.batches.append((self.data_tensors, self.metadata, start,i))
                count = 0
                ex = 0
                start = i
            count += avg_len
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
            copy = self.metadata[start:end].tolist()
            random.shuffle(copy)
            self.metadata[start:end] = torch.tensor(copy, dtype=torch.int32)
        # Form batches
        self.batches = []
        for i in range(self.metadata.shape[0]):
            self.batches.append((self.data_tensors, self.metadata, i, i+1))
        random.shuffle(self.batches)
        print(time_now(), 'Batching DONE: got {} batches; discarded {} examples ({} tokens)'.format(len(self.batches), 0, 0))

    def __len__(self):
        return len(self.batches) + (8 if self.randomized else 0)

    def __getitem__(self, idx):
        if self.device is None:
            self.device = torch.device('cuda:%d' % self.rank)
        batch_idx = idx % len(self.batches)
        if (batch_idx == 0) and self.randomized:
            self.shuffle_buckets_and_mark_batches()
        batch_idx = idx % len(self.batches)
        batch_item = self.batches[batch_idx]
        # return batch_item
        if self.predict_affixes:
            return en2kin_copy_batch_item_to_device(kinmt_en2kin_data_collate_fn([batch_item]), self.device)
        else:
            return kin2en_copy_batch_item_to_device(kinmt_kin2en_data_collate_fn([batch_item]), self.device)

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

def generic_item_batch(batch_item, predict_affixes=False):
    # Configurable, but keep it this way now
    max_seq_len = 512
    (data_tensors, metadata, start, end) = batch_item
    (global_lm_morphs,
     global_pos_tags,
     global_affixes,
     global_tokens_lengths,
     global_stems,
     global_en_tokens) = data_tensors

    batch_lm_morphs = []
    batch_pos_tags = []
    batch_affixes = []
    batch_tokens_lengths = []
    batch_stems = []

    batch_input_sequence_lengths = []

    batch_eng_tokens = []
    batch_eng_tokens_sequence_lengths = []

    # for ix in range(start,end):
    #     # avg_len =   metadata[ix,0].item()
    #     start_rw =  metadata[ix,1].item()
    #     end_rw =    metadata[ix,2].item()
    #     afx_start = metadata[ix,3].item()
    #     afx_end =   metadata[ix,4].item()
    #     start_en =  metadata[ix,5].item()
    #     end_en =    metadata[ix,6].item()
    for list_item in metadata[start:end].tolist():
        (avg_len, start_rw, end_rw, afx_start, afx_end, start_en, end_en) = tuple(list_item)
        batch_lm_morphs.append(global_lm_morphs[start_rw:end_rw])
        batch_pos_tags.append(global_pos_tags[start_rw:end_rw])
        batch_tokens_lengths.extend(global_tokens_lengths[start_rw:end_rw].tolist())
        batch_stems.append(global_stems[start_rw:end_rw])

        batch_affixes.append(global_affixes[afx_start:afx_end])

        batch_eng_tokens.append(global_en_tokens[start_en:end_en])

        batch_input_sequence_lengths.append(end_rw - start_rw)
        batch_eng_tokens_sequence_lengths.append(end_en - start_en)

    # Needed to fix decoding start bug
    if (sum(batch_tokens_lengths) == 0) and (len(batch_tokens_lengths)>0):
        batch_affixes = [torch.tensor([0],dtype=torch.int32)]
        batch_tokens_lengths[-1] = 1
    tokens_lengths = batch_tokens_lengths
    input_sequence_lengths = batch_input_sequence_lengths

    english_input_ids = torch.cat(batch_eng_tokens)
    english_sequence_lengths = batch_eng_tokens_sequence_lengths

    lm_morphs = torch.cat(batch_lm_morphs)
    pos_tags = torch.cat(batch_pos_tags)
    affixes = torch.cat(batch_affixes)
    stems = torch.cat(batch_stems)

    if predict_affixes:
        affixes_list = affixes.tolist()
        pred_affixes_list = [affixes_list[x - y: x] for x, y in zip(accumulate(batch_tokens_lengths), batch_tokens_lengths)]
        afx_prob = torch.zeros(len(pred_affixes_list), global_cfg.tot_num_affixes, dtype=torch.float)
        for i,lst in enumerate(pred_affixes_list):
            if (len(lst) > 0):
                afx_prob[i,lst] = 1.0
        affixes_prob = afx_prob
    else:
        affixes_prob = None

    copy_tokens_prob = None

    # Needed to fix decoding start bug
    # if len(affixes) == 0:
    #     affixes.append(0)
    #     tokens_lengths[-1] = 1
    afx = affixes.split(tokens_lengths)
    # [[2,4,5], [6,7]]
    afx_padded = pad_sequence(afx, batch_first=False, padding_value=0) # 0-pad because it goes to embedding layer
    afx_padded = afx_padded
    # afx_padded: (M,L), M: max morphological length

    m_masks = [torch.zeros((x + 4), dtype=torch.bool) for x in tokens_lengths]
    m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1)  # Shape: (L, 4+M) # 1-pad for src_key_padding

    data_item = (english_input_ids, english_sequence_lengths,
                 lm_morphs, pos_tags, stems, input_sequence_lengths,
                 afx_padded, m_masks_padded,
                 affixes_prob, tokens_lengths,
                 copy_tokens_prob)
    return data_item

def pick_first_collate_fn(batch_items):
    return batch_items[0]

def kinmt_process_batch(batch_items, copy_tokens_vocab_size, copy_kinya=False, predict_affixes=False, copy_at_all = False):
    return generic_item_batch(batch_items[0], predict_affixes=predict_affixes)

def kinmt_en2kin_data_collate_fn(batch_items):
    roberta_vocab_size = 50265
    data_item = kinmt_process_batch(batch_items, roberta_vocab_size, copy_kinya=False, predict_affixes=True)
    (english_input_ids, english_sequence_lengths,
     lm_morphs, pos_tags, stems, input_sequence_lengths,
     afx_padded, m_masks_padded,
     affixes_prob, tokens_lengths,
     copy_tokens_prob) = data_item

    seq_len = max(input_sequence_lengths)
    src_key_padding_mask = generate_input_key_padding_mask(english_sequence_lengths, ignore_last=False)
    tgt_key_padding_mask = generate_input_key_padding_mask(input_sequence_lengths, ignore_last=True)
    decoder_mask = generate_square_subsequent_mask(seq_len)

    batch_data_item = (english_input_ids, english_sequence_lengths,
                       lm_morphs, pos_tags, stems, input_sequence_lengths,
                       afx_padded, m_masks_padded,
                       affixes_prob, tokens_lengths,
                       copy_tokens_prob,
                       src_key_padding_mask, tgt_key_padding_mask, decoder_mask)
    return batch_data_item

def kinmt_kin2en_data_collate_fn(batch_items):
    data_item = kinmt_process_batch(batch_items, global_cfg.tot_num_stems, copy_kinya=True, predict_affixes=False)
    (english_input_ids, english_sequence_lengths,
     lm_morphs, pos_tags, stems, input_sequence_lengths,
     afx_padded, m_masks_padded,
     affixes_prob, tokens_lengths,
     copy_tokens_prob) = data_item

    seq_len = max(english_sequence_lengths)
    src_key_padding_mask = generate_input_key_padding_mask(input_sequence_lengths, ignore_last=False)
    tgt_key_padding_mask = generate_input_key_padding_mask(english_sequence_lengths, ignore_last=True)
    decoder_mask = generate_square_subsequent_mask(seq_len)

    batch_data_item = (english_input_ids, english_sequence_lengths,
                       lm_morphs, pos_tags, stems, input_sequence_lengths,
                       afx_padded, m_masks_padded,
                       affixes_prob, tokens_lengths,
                       copy_tokens_prob,
                       src_key_padding_mask, tgt_key_padding_mask, decoder_mask)
    return batch_data_item
