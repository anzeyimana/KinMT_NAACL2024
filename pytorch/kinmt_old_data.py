import random
from datetime import datetime
from itertools import accumulate
import os

import numpy as np
import torch
from morphoy import ParsedMorphoSentence, BOS_ID, EOS_ID
from modules import BaseConfig
from misc_functions import read_lines, time_now
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

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

def prepare_kinya_sentence_data_old(sentence: ParsedMorphoSentence, max_seq_len):
    lm_morphs = []
    pos_tags = []
    stems = []
    affixes = []
    tokens_lengths = []
    kinya_bpe_dict = dict()

    for token in sentence.tokens:
        lm_morphs.append(token.lm_morph_id)
        pos_tags.append(token.pos_tag_id)
        if token.uses_bpe:
            stems.append((token.stem_id, token.surface_form))
        else:
            stems.append((token.stem_id,None))
        affixes.extend(token.affixes)
        tokens_lengths.append(len(token.affixes))
        for tid in token.extra_tokens_ids:
            lm_morphs.append(token.lm_morph_id)
            pos_tags.append(token.pos_tag_id)
            stems.append((tid,None))
            affixes.extend(token.affixes)
            tokens_lengths.append(len(token.affixes))
        if token.uses_bpe:
            kinya_bpe_dict[token.raw_surface_form] = [token.stem_id] + token.extra_tokens_ids

    lm_morphs = [BOS_ID] + lm_morphs + [EOS_ID]
    pos_tags = [BOS_ID] + pos_tags + [EOS_ID]
    stems = [(BOS_ID,None)] + stems + [(EOS_ID,None)]
    tokens_lengths = [0] + tokens_lengths + [0]

    if len(stems) > max_seq_len:
        lm_morphs = [BOS_ID] + lm_morphs[1:(max_seq_len-1)] + [EOS_ID]
        pos_tags = [BOS_ID] + pos_tags[1:(max_seq_len-1)] + [EOS_ID]
        stems = [(BOS_ID,None)] + stems[1:(max_seq_len-1)] + [(EOS_ID,None)]
        tokens_lengths = [0] + tokens_lengths[1:(max_seq_len-1)] + [0]

    return (lm_morphs,
            pos_tags,
            affixes,
            tokens_lengths,
            stems,
            kinya_bpe_dict)

def prepare_kinya_sentence_data(sentence: ParsedMorphoSentence, max_seq_len):
    lm_morphs = []
    pos_tags = []
    stems = []
    affixes = []
    tokens_lengths = []
    kinya_bpe_dict = dict()

    for token in sentence.tokens:
        lm_morphs.append(token.lm_morph_id)
        pos_tags.append(token.pos_tag_id)
        if token.uses_bpe:
            stems.append((token.stem_id, token.surface_form))
        else:
            stems.append((token.stem_id,None))
        affixes.extend(token.affixes)
        tokens_lengths.append(len(token.affixes))
        for tid in token.extra_tokens_ids:
            lm_morphs.append(token.lm_morph_id)
            pos_tags.append(token.pos_tag_id)
            stems.append((tid,None))
            affixes.extend(token.affixes)
            tokens_lengths.append(len(token.affixes))
        # if token.uses_bpe:
        #     kinya_bpe_dict[token.raw_surface_form] = [token.stem_id] + token.extra_tokens_ids

    lm_morphs = [BOS_ID] + lm_morphs + [EOS_ID]
    pos_tags = [BOS_ID] + pos_tags + [EOS_ID]
    stems = [(BOS_ID,None)] + stems + [(EOS_ID,None)]
    tokens_lengths = [0] + tokens_lengths + [0]

    if len(stems) > max_seq_len:
        lm_morphs = [BOS_ID] + lm_morphs[1:(max_seq_len-1)] + [EOS_ID]
        pos_tags = [BOS_ID] + pos_tags[1:(max_seq_len-1)] + [EOS_ID]
        stems = [(BOS_ID,None)] + stems[1:(max_seq_len-1)] + [(EOS_ID,None)]
        tokens_lengths = [0] + tokens_lengths[1:(max_seq_len-1)] + [0]

    return (lm_morphs,
            pos_tags,
            affixes,
            tokens_lengths,
            stems,
            kinya_bpe_dict)

def prepare_english_sentence_data_old(line, max_seq_len):
    parsed_tokens = [EnglishToken(tk) for tk in line.split('\t')]
    engl_bpe_dict = dict()
    tokens = []
    for token in parsed_tokens:
        tokens.extend([((tid,token.surface_form) if ((i==0) and token.uses_bpe) else (tid,None)) for i,tid in enumerate(token.token_ids)])
        if token.uses_bpe:
            engl_bpe_dict[token.surface_form] = token.token_ids
    if len(tokens) > max_seq_len:
        tokens = [tokens[0]] + tokens[1:(max_seq_len-1)] + [tokens[-1]]
    return tokens, engl_bpe_dict

def prepare_english_sentence_data(line, max_seq_len):
    engl_bpe_dict = dict()
    tokens = [(int(tk),None) for tk in line.split('\t')]
    if len(tokens) > max_seq_len:
        tokens = [tokens[0]] + tokens[1:(max_seq_len-1)] + [tokens[-1]]
    return tokens, engl_bpe_dict

def read_parallel_data_files(data_dir, keywords, max_seq_len, english_ext='_en.txt'):
    data = []
    for keyword in keywords:
        rw_lines = read_lines(data_dir + '/parsed_' + keyword + '_rw.txt')
        en_lines = read_lines(data_dir + '/parsed_' + keyword + english_ext)
        assert len(rw_lines)==len(en_lines), "Mismatch number of lines @ {} > rw: {} vs en: {}".format(keyword, len(rw_lines), len(en_lines))
        data.extend([(((rw.count(';')+en.count('\t'))//2)+1 , rw , en) for rw,en in zip(rw_lines,en_lines)])
        # rw_data = [prepare_kinya_sentence_data(ParsedMorphoSentence(line, delimeter=';'), max_seq_len) for line in rw_lines]
        # en_data = [prepare_english_sentence_data(line, max_seq_len) for line in en_lines]
        # assert len(rw_data)==len(en_data), "Mismatch number of data lines @ {} > rw: {} vs en: {}".format(keyword, len(rw_data), len(en_data))
        # data.extend([( ((len(rw[0]) + len(en[0]))//2) , rw , en) for rw,en in zip(rw_data,en_data)])
    return data

global_cfg = BaseConfig()

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

    def __init__(self, data,
                 data_dir="kinmt/parallel_data_2022/txt",
                 max_seq_len=512,
                 english_ext='_en.txt',
                 max_tokens_per_batch=25000, num_shuffle_buckets=200,
                 randomized = False):
        # 1. Sort
        self.max_tokens_per_batch = max_tokens_per_batch
        self.randomized = randomized
        print(time_now(), 'Dataset reading ... from:', data)
        self.data = read_parallel_data_files(data_dir, data, max_seq_len, english_ext=english_ext)
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

def kinmt_process_batch(batch_items, copy_tokens_vocab_size, copy_kinya=False, predict_affixes=False, copy_at_all = False):
    # Configurable, but keep it this way now
    max_seq_len = 512
    
    (data, start, end) = batch_items[0]
    batch_lm_morphs = []
    batch_pos_tags = []
    batch_affixes = []
    batch_tokens_lengths = []
    batch_stems = []

    batch_input_sequence_lengths = []

    batch_eng_tokens = []
    batch_eng_tokens_sequence_lengths = []

    batch_copy_tokens = []
    batch_copy_tokens_lengths = []

    for (_,rw_line,en_line) in data[start:end]:
        (seq_lm_morphs,
         seq_pos_tags,
         seq_affixes,
         seq_tokens_lengths,
         seq_stems,
         kinya_bpe_dict) = prepare_kinya_sentence_data(ParsedMorphoSentence(rw_line, delimeter=';'), max_seq_len)

        (en_tokens,engl_bpe_dict) = prepare_english_sentence_data(en_line, max_seq_len)

        if copy_at_all:
            if copy_kinya:
                for (tid,tk) in en_tokens:
                    if tk is None:
                        batch_copy_tokens_lengths.append(0)
                    else:
                        if tk in kinya_bpe_dict:
                            cids = kinya_bpe_dict[tk]
                            batch_copy_tokens.extend(cids)
                            batch_copy_tokens_lengths.append(len(cids))
                        else:
                            batch_copy_tokens_lengths.append(0)
            else:
                for (sid, tk) in seq_stems:
                    if tk is None:
                        batch_copy_tokens_lengths.append(0)
                    else:
                        if tk in engl_bpe_dict:
                            cids = engl_bpe_dict[tk]
                            batch_copy_tokens.extend(cids)
                            batch_copy_tokens_lengths.append(len(cids))
                        else:
                            batch_copy_tokens_lengths.append(0)
        batch_eng_tokens.extend([tid for (tid,tk) in en_tokens])
        batch_eng_tokens_sequence_lengths.append(len(en_tokens))

        batch_lm_morphs.extend(seq_lm_morphs)
        batch_pos_tags.extend(seq_pos_tags)

        batch_affixes.extend(seq_affixes)

        batch_tokens_lengths.extend(seq_tokens_lengths)
        batch_stems.extend([sid for (sid,tk) in seq_stems])

        batch_input_sequence_lengths.append(len(seq_tokens_lengths))
    # Needed to fix decoding start bug
    if len(batch_affixes) == 0:
        batch_affixes.append(0)
        batch_tokens_lengths[-1] = 1
    tokens_lengths = batch_tokens_lengths
    input_sequence_lengths = batch_input_sequence_lengths

    english_input_ids = torch.tensor(batch_eng_tokens)
    english_sequence_lengths = batch_eng_tokens_sequence_lengths

    lm_morphs = torch.tensor(batch_lm_morphs)
    pos_tags = torch.tensor(batch_pos_tags)
    affixes = torch.tensor(batch_affixes)
    stems = torch.tensor(batch_stems)

    if predict_affixes:
        pred_affixes_list = [batch_affixes[x - y: x] for x, y in zip(accumulate(batch_tokens_lengths), batch_tokens_lengths)]
        afx_prob = torch.zeros(len(pred_affixes_list), global_cfg.tot_num_affixes, dtype=torch.float)
        for i,lst in enumerate(pred_affixes_list):
            if (len(lst) > 0):
                afx_prob[i,lst] = 1.0
        affixes_prob = afx_prob
    else:
        affixes_prob = None

    if copy_at_all:
        pred_copy_tokens_list = [batch_copy_tokens[x - y: x] for x, y in zip(accumulate(batch_copy_tokens_lengths), batch_copy_tokens_lengths)]
        copy_tk_prob = torch.zeros(len(pred_copy_tokens_list), copy_tokens_vocab_size, dtype=torch.float)
        for i,lst in enumerate(pred_copy_tokens_list):
            if (len(lst) > 0):
                copy_tk_prob[i,lst] = 1.0
        copy_tokens_prob = copy_tk_prob
    else:
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
