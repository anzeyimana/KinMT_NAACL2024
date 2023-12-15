from __future__ import print_function, division

import gc
import math
import os
import os.path
import os.path
import sys
import time
from itertools import accumulate

import progressbar
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from morphoy import parse_text_to_morpho_sentence, ParsedMorphoSentence
from arguments import py_trainer_args
from kinmt_data import prepare_kinya_sentence_data, generate_input_key_padding_mask
from kinmt_kin2en import Kin2EnTransformer
from mybert import MyBERT_from_pretrained, MyBERT_large_from_pretrained
from modules import BaseConfig
from misc_functions import read_lines
from misc_functions import time_now
from fairseq.models.transformer_lm import TransformerLanguageModel

SORT_DECODE_TIME = 0.0
GPU_TIME = 0.0
TRANSLATE_TIME = 0.0
TRANSLATION_SENTENCE_COUNT = 0.0
TRANSLATION_TOKEN_COUNT = 0.0
EXPANSION_TIME = 0.0

def reset_and_print_kin2en_latency_info():
    global SORT_DECODE_TIME
    global GPU_TIME
    global TRANSLATE_TIME
    global EXPANSION_TIME
    global TRANSLATION_SENTENCE_COUNT
    global TRANSLATION_TOKEN_COUNT

    PRE_POST_PROC_TIME = TRANSLATE_TIME - (GPU_TIME + EXPANSION_TIME + SORT_DECODE_TIME)
    if (TRANSLATION_SENTENCE_COUNT > 0):
        print('SENTENCE STATS:')
        print('Translation/sentence:   {:.3f} sec - {:.0f} %'.format(TRANSLATE_TIME/TRANSLATION_SENTENCE_COUNT, 100.0 * TRANSLATE_TIME / TRANSLATE_TIME))
        print('Sort-Decode/sentence: {:.3f} sec - {:.0f} %'.format(SORT_DECODE_TIME/TRANSLATION_SENTENCE_COUNT, 100.0 * SORT_DECODE_TIME / TRANSLATE_TIME))
        print('GPU/sentence:           {:.3f} sec - {:.0f} %'.format(GPU_TIME/TRANSLATION_SENTENCE_COUNT, 100.0 * GPU_TIME / TRANSLATE_TIME))
        print('Expansion/sentence:     {:.3f} sec - {:.0f} %'.format(EXPANSION_TIME/TRANSLATION_SENTENCE_COUNT, 100.0 * EXPANSION_TIME / TRANSLATE_TIME))
        print('PrePostProc/sentence:   {:.3f} sec - {:.0f} %'.format(PRE_POST_PROC_TIME/TRANSLATION_SENTENCE_COUNT, 100.0 * PRE_POST_PROC_TIME / TRANSLATE_TIME))

        print('TOKEN STATS:')
        print('Translation/token:   {:.3f} msec - {:.0f} %'.format(1000.0 * TRANSLATE_TIME/TRANSLATION_TOKEN_COUNT, 100.0 * TRANSLATE_TIME / TRANSLATE_TIME))
        print('Sort-Decode/token: {:.3f} msec - {:.0f} %'.format(1000.0 * SORT_DECODE_TIME/TRANSLATION_TOKEN_COUNT, 100.0 * SORT_DECODE_TIME / TRANSLATE_TIME))
        print('GPU/token:           {:.3f} msec - {:.0f} %'.format(1000.0 * GPU_TIME/TRANSLATION_TOKEN_COUNT, 100.0 * GPU_TIME / TRANSLATE_TIME))
        print('Expansion/token:     {:.3f} msec - {:.0f} %'.format(1000.0 * EXPANSION_TIME/TRANSLATION_TOKEN_COUNT, 100.0 * EXPANSION_TIME / TRANSLATE_TIME), flush=True)
        print('PrePostProc/token:   {:.3f} msec - {:.0f} %'.format(1000.0 * PRE_POST_PROC_TIME/TRANSLATION_TOKEN_COUNT, 100.0 * PRE_POST_PROC_TIME / TRANSLATE_TIME), flush=True)

    SORT_DECODE_TIME = 0.0
    GPU_TIME = 0.0
    TRANSLATE_TIME = 0.0
    TRANSLATION_SENTENCE_COUNT = 0.0
    TRANSLATION_TOKEN_COUNT = 0.0
    EXPANSION_TIME = 0.0


def reset_latency_info():
    global SORT_DECODE_TIME
    global GPU_TIME
    global TRANSLATE_TIME
    global EXPANSION_TIME
    global TRANSLATION_SENTENCE_COUNT
    global TRANSLATION_TOKEN_COUNT

    SORT_DECODE_TIME = 0.0
    GPU_TIME = 0.0
    TRANSLATE_TIME = 0.0
    TRANSLATION_SENTENCE_COUNT = 0.0
    TRANSLATION_TOKEN_COUNT = 0.0
    EXPANSION_TIME = 0.0


def read_shared_input_data():
    metadata = []
    global_lm_morphs = []
    global_pos_tags = []
    global_affixes = []
    global_tokens_lengths = []
    global_stems = []
    start_rw = 0
    afx_start = 0
    print(time_now(), f'Reading input file ...', flush=True)
    rw_lines = read_lines('kinmt/parallel_data_2022/txt/parsed_morpho_corpus_sentences_clean_2023-07-27_rw.txt')
    print(time_now(), f'Appending data ==> {len(rw_lines)} lines ...', flush=True)
    itr = 0
    with progressbar.ProgressBar(initial_value=0,
                                 max_value=len(rw_lines),
                                 redirect_stdout=True) as bar:
        for rw_line in rw_lines:
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
             seq_stems) = prepare_kinya_sentence_data(ParsedMorphoSentence(rw_line, delimeter=';'), 500)
            rw_len = len(seq_stems)

            global_lm_morphs.extend(seq_lm_morphs)
            global_pos_tags.extend(seq_pos_tags)
            global_affixes.extend(seq_affixes)
            global_tokens_lengths.extend(seq_tokens_lengths)
            global_stems.extend(seq_stems)
            end_rw = start_rw + rw_len
            afx_end = afx_start + len(seq_affixes)
            metadata.append((start_rw, end_rw, afx_start, afx_end))
            start_rw = end_rw
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
    gc.collect()
    # Put tensors into shared memory
    global_lm_morphs.share_memory_()
    global_pos_tags.share_memory_()
    global_affixes.share_memory_()
    global_tokens_lengths.share_memory_()
    global_stems.share_memory_()
    # Put tensors together
    global_tensors = (global_lm_morphs, global_pos_tags, global_affixes, global_tokens_lengths, global_stems)
    return global_tensors, metadata

def input_data_item(global_tensors, metadata_item, device):
    (global_lm_morphs,
     global_pos_tags,
     global_affixes,
     global_tokens_lengths,
     global_stems) = global_tensors
    (start_rw, end_rw, afx_start, afx_end) = metadata_item

    (lm_morphs,
     pos_tags,
     affixes,
     tokens_lengths,
     stems) = (global_lm_morphs[start_rw:end_rw].to(device, dtype=torch.long),
                     global_pos_tags[start_rw:end_rw].to(device, dtype=torch.long),
                     global_affixes[afx_start:afx_end].to(dtype=torch.long),
                     global_tokens_lengths[start_rw:end_rw].tolist(),
                     global_stems[start_rw:end_rw].to(device, dtype=torch.long))

    input_sequence_lengths = [len(tokens_lengths)]

    src_key_padding_mask = generate_input_key_padding_mask(input_sequence_lengths, ignore_last=False).to(device)

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
    return seed_data_item, None


def kin2en_beam_search_init(tgt_BOS, device):
    tgt_input = torch.tensor([tgt_BOS], dtype=torch.long, device=device)
    tgt_input_lengths = [1]

    tgt_input_log_prob = [0.0]

    tgt_complete_sequences = []
    tgt_complete_sequences_prob = []

    return (tgt_input, tgt_input_lengths,
            tgt_input_log_prob,
            tgt_complete_sequences, tgt_complete_sequences_prob)


def find_kinya_copy_tokens(kinya_bpe_dict, tgt_copy_prob, english_decoder, copy_threshold=0.5, intersect_threshold=0.5):
    eng_lm_BOS_idx = 0
    eng_lm_EOS_idx = 2
    log_eps = 1e-36
    N, V = tgt_copy_prob.shape
    extensions = []
    for n in range(N):
        ext = []
        ext_token = None
        ext_prob = 0.0
        tot_prob = tgt_copy_prob[n, :].sum().item()
        copy_idx = (tgt_copy_prob[n, :] > copy_threshold).nonzero(as_tuple=True)[0].tolist()
        for key in kinya_bpe_dict:
            src_idx = kinya_bpe_dict[key]
            shared = set(copy_idx).intersection(set(src_idx))
            shared_ratio = (1.0 * len(shared)) / (1.0 * len(src_idx))
            if shared_ratio > intersect_threshold:
                copy_prob = tgt_copy_prob[n, src_idx].sum().item() / tot_prob
                if copy_prob > ext_prob:
                    ext_prob = copy_prob
                    ext_token = key
        if ext_token is not None:
            token_ids = english_decoder.encode(ext_token).cpu().numpy().tolist()
            if token_ids[0] == eng_lm_BOS_idx:
                token_ids = token_ids[1:]
            if token_ids[-1] == eng_lm_EOS_idx:
                token_ids = token_ids[:-1]
            ext = token_ids
        extensions.append((ext, math.log(ext_prob + log_eps), ext_token))
    return extensions

def copy_to_completed(tgt_complete_sequences,  tgt_complete_sequences_prob,
                      out_seq, src_attn_weights, batch,
                      pending_seq_prob, token_prob):
    tgt_complete_sequences.append(out_seq)
    new_sentence_prob = pending_seq_prob + token_prob
    # Length normalization
    alpha = 0.2
    beta = 0.8
    gamma = 0.2
    length_penalty = math.pow(((out_seq.size(0) + 5.0) / 6.0), alpha)
    coverage_penalty = sum([math.log(min(src_attn_weights[batch, :, ss].sum().item(), 1.0)) for ss in
                            range(src_attn_weights.size(2))]) * beta
    eos_penalty = (gamma * src_attn_weights.size(2)) / src_attn_weights.size(1)
    new_sentence_prob = (new_sentence_prob / length_penalty) + coverage_penalty + eos_penalty
    tgt_complete_sequences_prob.append(new_sentence_prob)


def kin2en_beam_search_expand(tgt_input, tgt_input_lengths,
                              tgt_input_log_prob,
                              tgt_top_logits, tgt_top_idx,  # (N,beam_size)
                              tgt_EOS,
                              device, max_batch_size,
                              tgt_complete_sequences, tgt_complete_sequences_prob,
                              src_attn_weights,
                              kinya_bpe_dict, tgt_copy_prob, english_decoder, copy_threshold=0.5,
                              intersect_threshold=0.5,
                              max_seq_length=512):
    next_tgt_inputs = []
    next_tgt_inputs_prob = []
    # src_attn_weights: N,T,S
    extensions = None
    if tgt_copy_prob is not None:
        extensions = find_kinya_copy_tokens(kinya_bpe_dict, tgt_copy_prob, english_decoder,
                                            copy_threshold=copy_threshold,
                                            intersect_threshold=intersect_threshold)
    tgt_input_boundaries = [(x - y, x) for x, y in zip(accumulate(tgt_input_lengths), tgt_input_lengths)]
    for n in range(tgt_top_logits.size(0)):  # n: batch
        (inp_start, inp_end) = tgt_input_boundaries[n]
        ext = []
        ext_lprob = -999999.99
        if tgt_copy_prob is not None:
            (ext, ext_lprob, ext_token) = extensions[n]
        lprob = F.log_softmax(tgt_top_logits[n, :],
                              dim=-1).cpu().detach().numpy()  # shape: (beam_size), normalizing by top BS outputs
        if len(ext) > 0:
            out_seq = torch.cat((tgt_input[inp_start:inp_end], torch.tensor(ext, device=device)), dim=0)
            if (out_seq.size(0) >= max_seq_length):
                    copy_to_completed(tgt_complete_sequences, tgt_complete_sequences_prob, out_seq, src_attn_weights, n, tgt_input_log_prob[n], ext_lprob)
            else:
                next_tgt_inputs.append(out_seq)
                # TODO: not sure to multiply lprob by number of extensions or just add it once to match other hypotheses so far
                next_tgt_inputs_prob.append(tgt_input_log_prob[n] + ext_lprob)  # + (ext_lprob * len(ext)))
        else:
            for k in range(tgt_top_logits.size(1)):  # k: target token @ beam dimension
                out_seq = torch.cat((tgt_input[inp_start:inp_end], torch.tensor([tgt_top_idx[n, k]], device=device)), dim=0)
                if (tgt_top_idx[n, k].item() == tgt_EOS) or (out_seq.size(0) >= max_seq_length):
                    copy_to_completed(tgt_complete_sequences, tgt_complete_sequences_prob, out_seq, src_attn_weights, n, tgt_input_log_prob[n], lprob[k])
                else:
                    next_tgt_inputs.append(out_seq)
                    next_tgt_inputs_prob.append(tgt_input_log_prob[n] + lprob[k])

    batch_indices = sorted(range(len(next_tgt_inputs_prob)), key=lambda i: next_tgt_inputs_prob[i], reverse=True)[
                    :max_batch_size]
    tgt_input = None
    tgt_input_lengths = []
    tgt_input_prob = []
    for n in batch_indices:
        seq = next_tgt_inputs[n]
        prob = next_tgt_inputs_prob[n]
        tgt_input = seq if (tgt_input is None) else torch.cat((tgt_input, seq), dim=0)
        tgt_input_lengths.append(seq.size(0))
        tgt_input_prob.append(prob)
    return (tgt_input, tgt_input_lengths,
            tgt_input_prob,
            tgt_complete_sequences, tgt_complete_sequences_prob)


def kin2en_decode(kin2en_model,
                  english_lm,
                  english_decoder,
                  device,
                  enc_output, enc_attn_bias, bert_src, src_key_padding_mask,
                  beam_size,
                  max_eval_sequences,
                  kinya_bpe_dict,
                  tgt_BOS=0, tgt_EOS=2,
                  max_text_length=100,
                  copy_threshold=0.5, intersect_threshold=0.5):
    global GPU_TIME
    global EXPANSION_TIME
    (tgt_input, tgt_input_lengths, tgt_input_log_prob,
     tgt_complete_sequences, tgt_complete_sequences_prob) = kin2en_beam_search_init(tgt_BOS, device)
    total_length = 0
    while True:
        with torch.no_grad():
            ttt = time.time()
            (tgt_out, tgt_copy_prob), src_attn_weights = kin2en_model.decode_en(english_lm,
                                                                                enc_output, enc_attn_bias, bert_src,
                                                                                src_key_padding_mask,
                                                                                tgt_input, tgt_input_lengths)
            GPU_TIME += (time.time() - ttt)
        qqq = time.time()
        tgt_top_logits, tgt_top_idx = tgt_out.topk(beam_size, dim=-1)  # (N,beam_size)
        # tgt_top_logits = tgt_top_logits[-1,:,:] # (N,beam_size)
        # tgt_top_idx = tgt_top_idx[-1,:,:] # (N,beam_size)
        # TODO: Find top potential copies from src ==> tgt_copy_prob: (L,N,|V_src|)
        # TODO: Get copy dictionary from src
        (tgt_input, tgt_input_lengths,
         tgt_input_log_prob,
         tgt_complete_sequences,
         tgt_complete_sequences_prob) = kin2en_beam_search_expand(tgt_input, tgt_input_lengths, tgt_input_log_prob,
                                                                  tgt_top_logits, tgt_top_idx,  # (L,N,beam_size)
                                                                  tgt_EOS,
                                                                  device, beam_size,
                                                                  tgt_complete_sequences, tgt_complete_sequences_prob,
                                                                  src_attn_weights,
                                                                  kinya_bpe_dict, tgt_copy_prob, english_decoder,
                                                                  copy_threshold=copy_threshold,
                                                                  intersect_threshold=intersect_threshold)
        EXPANSION_TIME += (time.time() - qqq)
        if (len(tgt_complete_sequences) >= max_eval_sequences) or (tgt_input is None):
            return (tgt_input, tgt_input_lengths,
                    tgt_input_log_prob,
                    tgt_complete_sequences, tgt_complete_sequences_prob)
        if (total_length > max_text_length):
            return (tgt_input, tgt_input_lengths,
                    tgt_input_log_prob,
                    tgt_complete_sequences, tgt_complete_sequences_prob)
        total_length += 1


def parse_kinya_sentence(input_sentence: str, ffi, lib, device, cfg: BaseConfig):
    predict_affixes = False
    if lib is not None:
        parsed_sentence = parse_text_to_morpho_sentence(ffi, lib, input_sentence)
    else:
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
        afx_prob = torch.zeros(len(pred_affixes_list), cfg.tot_num_affixes)
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
    return seed_data_item, None

def english_normalization(sent):
    if sent is not None:
        if len(sent) > 0:
            sent = ' '.join(sent.replace('\n',' ').replace('\t',' ').split())
            sent = sent.replace(' <unk> s ', '\'s ')
            sent = sent.replace(' <unk> s.', '\'s.')
            sent = sent.replace(' <unk> s?', '\'s?')
            sent = sent.replace(' <unk> s!', '\'s!')
            sent = sent.replace(' <unk> s,', '\'s,')
            sent = sent.replace(' <unk> s;', '\'s;')
            sent = sent.replace('<unk>', ' ')
            sent = ' '.join(sent.replace('\n', ' ').replace('\t', ' ').split())
    return sent

def kin2en_translate_item(kin2en_model,
                          mybert_encoder,
                          english_decoder,
                          seed_data_item,
                          device,
                          beam_size, max_eval_sequences,
                          copy_threshold=0.5, intersect_threshold=0.5):
    global GPU_TIME
    global SORT_DECODE_TIME
    global TRANSLATE_TIME
    global TRANSLATION_SENTENCE_COUNT
    global TRANSLATION_TOKEN_COUNT
    global EXPANSION_TIME

    english_lm = None
    kinya_bpe_dict = None

    start_translate_time = time.time()

    len_tokens = seed_data_item[2].shape[0]
    max_text_length = int(math.ceil(len_tokens * 2.5))

    TRANSLATION_SENTENCE_COUNT += 1.0
    TRANSLATION_TOKEN_COUNT += (1.0*len_tokens)

    with torch.no_grad():
        ttt = time.time()
        (enc_output, enc_attn_bias,
         bert_src, src_key_padding_mask) = kin2en_model.encode_kin(mybert_encoder, seed_data_item)
        GPU_TIME += (time.time() - ttt)

    (tgt_input, tgt_input_lengths,
     tgt_input_log_prob,
     tgt_complete_sequences,
     tgt_complete_sequences_prob) = kin2en_decode(kin2en_model, english_lm, english_decoder, device,
                                                  enc_output, enc_attn_bias, bert_src, src_key_padding_mask,
                                                  beam_size, max_eval_sequences,
                                                  kinya_bpe_dict,
                                                  tgt_BOS=0, tgt_EOS=2, max_text_length=max_text_length,
                                                  copy_threshold=copy_threshold,
                                                  intersect_threshold=intersect_threshold)
    sss = time.time()
    complete_translations = []
    for prob, seq in zip(tgt_complete_sequences_prob, tgt_complete_sequences):
        complete_translations.append((prob, english_normalization(english_decoder.decode(seq))))
    complete_translations = sorted(complete_translations, reverse=True, key=lambda x: x[0])

    pending_translations = []
    if len(complete_translations) == 0:
        start = 0
        for prob, tgt_len in zip(tgt_input_log_prob, tgt_input_lengths):
            end = start + tgt_len
            pending_translations.append((prob, english_normalization(english_decoder.decode(tgt_input[start:end]))))
            start = end
        pending_translations = sorted(pending_translations, reverse=True, key=lambda x: x[0])

    SORT_DECODE_TIME += (time.time() - sss)

    TRANSLATE_TIME += (time.time() - start_translate_time)

    return complete_translations, pending_translations

def translate_one(global_tensors, metadata, idx, device, kin2en_model, mybert_encoder, english_decoder, beam_size, max_eval_sequences, copy_threshold, intersect_threshold, out_file):
    seed_data_item, _ = input_data_item(global_tensors, metadata[idx], device)
    (complete_translations,
     pending_translations) = kin2en_translate_item(kin2en_model,
                                                   mybert_encoder,
                                                   english_decoder,
                                                   seed_data_item,
                                                   device,
                                                   beam_size, max_eval_sequences,
                                                   copy_threshold=copy_threshold,
                                                   intersect_threshold=intersect_threshold)
    kinmt_translation = 'Nothing'
    if len(complete_translations) > 0:
        kinmt_translation = complete_translations[0][1]
    elif len(pending_translations) > 0:
        kinmt_translation = pending_translations[0][1]
    out_file.write(f'{kinmt_translation}\n')
    out_file.flush()


def shared_mem_kin2en_translate_range(kin2en_model,
                                      mybert_encoder,
                                      english_decoder,
                                      global_tensors, metadata, n, start_at, end_at, device,
                                      beam_size, max_eval_sequences,
                                      copy_threshold=0.5, intersect_threshold=0.5):
    output_filename = f'kinmt/parallel_data_2022/txt/morpho_corpus_sentences_clean_2023-07-27_{n}_{start_at}_{end_at}_en.txt'
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        if (n == 1):
            itr = 0
            with progressbar.ProgressBar(max_value=(end_at - start_at + 10), redirect_stdout=True) as bar:
                for idx in range(start_at-1, end_at):
                    if ((itr % 100) == 0):
                        reset_and_print_kin2en_latency_info()
                        bar.update(itr)
                    translate_one(global_tensors, metadata, idx, device, kin2en_model, mybert_encoder, english_decoder, beam_size, max_eval_sequences, copy_threshold, intersect_threshold, out_file)
                    itr += 1
        else:
            for idx in range(start_at-1, end_at):
                translate_one(global_tensors, metadata, idx, device, kin2en_model, mybert_encoder, english_decoder, beam_size, max_eval_sequences, copy_threshold, intersect_threshold, out_file)
                reset_latency_info()

def translation_process(n_1, procs, gpus, args, cfg, data):
    global_tensors, metadata = data
    n = n_1 + 1
    total_lines = len(metadata)
    idx = n - 1
    rank = idx % (gpus)
    batch = int(math.ceil(total_lines / procs))
    start_at = (idx * batch) + 1
    end_at = min((idx + 1) * batch, total_lines)

    device = torch.device('cuda:%d' % rank)
    torch.cuda.set_device(rank)
    home_path = args.home_path
    curr_save_file_path = home_path + f"data/{args.kinmt_model_name}.pt"

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
    english_decoder = TransformerLanguageModel.from_pretrained('wmt19.en', 'model.pt',
                                                               tokenizer='moses', bpe='fastbpe').cpu()
    english_decoder.float()
    english_decoder.eval()
    if args.kinmt_use_gpt:
        english_lm = english_decoder.models[0]
    else:
        del english_decoder.models[0]
        del english_decoder.models
        english_decoder.models = []
        english_lm = None
    kin2en_model = Kin2EnTransformer(args, cfg, mybert_encoder, english_lm,
                                     use_cross_pos_attn=args.use_cross_positional_attn_bias).to(device)
    kin2en_model.float()
    kin2en_model.eval()

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if args.load_saved_model:
        print(time_now(), f'Loading model state from {curr_save_file_path}', flush=True)
    kb_state_dict = torch.load(curr_save_file_path, map_location=map_location)
    kin2en_model.load_state_dict(kb_state_dict['model_state_dict'])
    epoch = kb_state_dict['epoch']
    best_valid_loss = kb_state_dict['best_valid_loss']
    steps = kb_state_dict['lr_scheduler_state_dict']['num_iters']
    print(time_now(), f'Trained epochs: {epoch}', flush=True)
    print(time_now(), f'Trained steps: {steps}', flush=True)
    print(time_now(), 'Best valid loss: {:.4f}'.format(best_valid_loss), flush=True)
    del kb_state_dict
    gc.collect()

    copy_threshold = 0.5
    intersect_threshold = 0.5
    beam_size = 4
    max_eval_sequences = 5

    shared_mem_kin2en_translate_range(kin2en_model,
                                      mybert_encoder,
                                      english_decoder,
                                      global_tensors, metadata, n, start_at, end_at, device,
                                      beam_size, max_eval_sequences,
                                      copy_threshold=copy_threshold, intersect_threshold=intersect_threshold)


def back_translate_main(procs, gpus):
    args = py_trainer_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    args.world_size = procs
    args.gpus = gpus

    cfg = BaseConfig()

    data = read_shared_input_data()

    args.load_saved_model = True
    args.kinmt_use_bert = True
    args.kinmt_use_gpt = False
    args.use_cross_positional_attn_bias = True
    args.kinmt_use_copy_loss = False
    args.kinmt_bert_large = True
    args.kinmt_model_name = "kin2en_base_large_bert_trial_backtrans_eval_xpos_2023-07-22"
    args.home_path = "/home/user/MORPHO/"

    mp.spawn(translation_process, nprocs=procs, args=(procs, gpus, args, cfg, data,))

if __name__ == '__main__':
    back_translate_main(procs=24, gpus=8)
