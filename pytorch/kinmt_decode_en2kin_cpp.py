import math
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from morpho_model import pos_tag_initials
from morphoy import build_morphoy_lib
from kinmt_data import generate_input_key_padding_mask, generate_square_subsequent_mask
from kinmt_en2kin import En2KinTransformer
from mygpt import MyGPT_from_pretrained
from modules import BaseConfig
from misc_functions import normalize_kinya_text, read_lines
from fairseq.models.roberta import RobertaModel

english_BOS_idx = 0
english_EOS_idx = 2

PAD_ID = 0
UNK_ID = 1
MSK_ID = 2
BOS_ID = 3
EOS_ID = 4

DECODE_TIME = 0.0
GPU_TIME = 0.0
TRANSLATE_TIME = 0.0
TRANSLATION_SENTENCE_COUNT = 0.0
TRANSLATION_TOKEN_COUNT = 0.0
EXPANSION_TIME = 0.0

def en2kin_init_models(args, rank=0, morpho_conf = 'config_morpho.conf'):
    build_morphoy_lib()
    from morphoy import ffi, lib

    cfg = BaseConfig()
    device = torch.device('cuda:%d' % rank)
    torch.cuda.set_device(rank)
    home_path = args.home_path
    curr_save_file_path = home_path + f"data/{args.kinmt_model_name}.pt"

    if args.kinmt_use_gpt:
        my_gpt = MyGPT_from_pretrained(args, cfg,
                                             'mygpt_final_2022-12-23_operated_full_base_2022-12-13.pt').to(
            device)
        my_gpt.float()
        my_gpt.eval()
        kin_lm_model = my_gpt.encoder
    else:
        kin_lm_model = None
    if args.kinmt_use_bert:
        engl_roberta_model = RobertaModel.from_pretrained('roberta.base',
                                                          checkpoint_file='model.pt').to(device)
        engl_roberta_model.float()
        engl_roberta_model.eval()
    else:
        engl_roberta_model = None

    en2kin_model = En2KinTransformer(args, cfg, engl_roberta_model, kin_lm_model,
                                     use_cross_pos_attn=args.use_cross_positional_attn_bias).to(device)
    en2kin_model.float()

    kb_state_dict = torch.load(curr_save_file_path, map_location=device)
    en2kin_model.load_state_dict(kb_state_dict['model_state_dict'])
    en2kin_model.eval()

    lib.klm_init(morpho_conf.encode('utf-8'))
    print('MORPHOY Morpho-Synthesis Lib Ready!', flush=True)

    return (en2kin_model, engl_roberta_model, kin_lm_model, ffi, lib, device)


def initial_input_outputs(english_sentence, engl_roberta_model, ENGL_SEGMENT_PER_TOKEN = False):
    english_BOS_idx = 0
    english_EOS_idx = 2

    line_ids = []
    line_ids.append(('', [english_BOS_idx]))

    if ENGL_SEGMENT_PER_TOKEN:
        tokens = ' '.join(english_sentence.replace('\t', ' ').split()).split()
        for token in tokens:
            token_ids = engl_roberta_model.encode(token).cpu().numpy().tolist()
            if token_ids[0] == english_BOS_idx:
                token_ids = token_ids[1:]
            if token_ids[-1] == english_EOS_idx:
                token_ids = token_ids[:-1]
            line_ids.append((token, token_ids))
        line_ids.append(('', [english_EOS_idx]))
        token_ids = [i for x, y in line_ids for i in y]
    else:
        tokens_line = ' '.join(english_sentence.replace('<unk>', ' ').replace('<s>', ' ').replace('</s>', ' ').replace('\t', ' ').split())
        token_ids = engl_roberta_model.encode(tokens_line).cpu().numpy().tolist()
        if token_ids[0] != english_BOS_idx:
            token_ids = [english_BOS_idx] + token_ids
        if token_ids[-1] != english_EOS_idx:
            token_ids = token_ids + [english_EOS_idx]

    # print(f'>> {english_sentence}')
    # back_engl = engl_roberta_model.decode(torch.tensor(token_ids))
    # print(f'<< {back_engl}')

    # engl_roberta_model.encode(english_sentence).cpu().numpy().tolist()

    if token_ids[0] != english_BOS_idx:
        token_ids = [english_BOS_idx] + token_ids
    if token_ids[-1] != english_EOS_idx:
        token_ids = token_ids + [english_EOS_idx]
    outputs = [([(BOS_ID, BOS_ID, BOS_ID, [], '<s>')], 0.0, False)]
    input_outputs = (token_ids, outputs)
    return input_outputs


def batch_data(input_outputs, device):
    (en_ids, outputs) = input_outputs
    batch_stems = []
    batch_pos_tags = []
    batch_lm_morphs = []
    batch_affixes = []
    tokens_lengths = []
    input_sequence_lengths = []
    for rw, prob, eos in outputs:
        for stem, pos, afset, affixes, tkn in rw:
            batch_stems.append(stem)
            batch_pos_tags.append(pos)
            batch_lm_morphs.append(afset)
            batch_affixes.extend(affixes)
            tokens_lengths.append(len(affixes))
        input_sequence_lengths.append(len(rw))

    if len(batch_affixes) == 0:
        batch_affixes = [0]
        tokens_lengths[-1] = 1

    english_input_ids = torch.tensor(en_ids).to(device)
    english_sequence_lengths = [len(en_ids)]

    lm_morphs = torch.tensor(batch_lm_morphs).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    affixes = torch.tensor(batch_affixes)  # .to(device)
    stems = torch.tensor(batch_stems).to(device)

    affixes_prob = None

    afx = affixes.split(tokens_lengths)
    if sum(tokens_lengths) == 0:
        afx = afx[1:] + (torch.tensor([0], dtype=torch.long),)
    # [[2,4,5], [6,7]]
    afx_padded = pad_sequence(afx, batch_first=False, padding_value=0)  # 0-pad because it goes to embedding layer
    afx_padded = afx_padded.to(device, dtype=torch.long)
    # afx_padded: (M,L), M: max morphological length
    # m_masks_padded = None
    # if afx_padded.nelement() > 0:
    m_masks = [torch.zeros((x + 4), dtype=torch.bool, device=stems.device) for x in tokens_lengths]
    m_masks_padded = pad_sequence(m_masks, batch_first=True,
                                  padding_value=1)  # Shape: (L, 4+M) # 1-pad for src_key_padding

    seq_len = max(input_sequence_lengths)
    src_key_padding_mask = generate_input_key_padding_mask(english_sequence_lengths, ignore_last=False).to(device)
    tgt_key_padding_mask = generate_input_key_padding_mask(input_sequence_lengths, ignore_last=False).to(device)
    decoder_mask = generate_square_subsequent_mask(seq_len).to(device)

    batch_data_item = (device, english_input_ids, english_sequence_lengths,
                       lm_morphs, pos_tags, stems, input_sequence_lengths,
                       afx_padded, m_masks_padded,
                       affixes_prob, tokens_lengths,
                       src_key_padding_mask, tgt_key_padding_mask, decoder_mask)

    return batch_data_item


def join_token(txt, space):
    if (txt[:2] == '@@'):
        # txt = '<'+txt[2:]+'>'
        txt = txt[2:]
        space = False
    elif (txt[:1] == '▁'):
        # txt = '<'+txt[1:]+'>'
        txt = txt[1:]
    return (' ' + txt) if space else txt


def decode_kinya_sequence(seq):
    ret = ''
    tag_dict = dict()
    for stem, pos, afset, affixes, txt in seq:
        tag_dict[txt] = pos_tag_initials(pos)
    for stem, pos, afset, affixes, txt in seq:
        if (txt != '<s>') and (txt != '</s>'):
            if len(ret) > 0:
                ret += join_token(txt, True)
            else:
                ret += join_token(txt, False)
    return normalize_kinya_text(ret, tag_dict=tag_dict)

def process_new_entry(bidx, seq, src_attn_weights, log_prob, entry, ffi):
    new_seq = []
    new_seq.extend(seq)
    affixes = []
    for i in range(entry.affixes_len):
        affixes.append(entry.affixes[i])
    new_seq.append((entry.stem, entry.pos_tag, entry.afset, affixes, ffi.string(entry.word).decode("utf-8")))
    new_prob = entry.prob
    if not np.isfinite(new_prob):
        new_prob = 1e-20
    if new_prob <= 0.0:
        new_prob = 1e-20
    new_prob = log_prob + math.log(new_prob)
    eos = (entry.stem == EOS_ID) and (entry.pos_tag == EOS_ID) and (entry.afset == EOS_ID)
    if eos:
        alpha = 0.2
        beta = 0.8
        gamma = 0.2
        length_penalty = math.pow(((len(new_seq) + 5.0) / 6.0), alpha)
        sum_S = 0.0
        for ss in range(src_attn_weights.size(2)):
            sum_T = 0.0
            for tt in range(src_attn_weights.size(1)):
                sum_T += src_attn_weights[bidx, tt, ss].item()
            sum_S += math.log(min(sum_T, 1.0))
        coverage_penalty = sum_S * beta
        eos_penalty = (gamma * src_attn_weights.size(2)) / src_attn_weights.size(1)
        new_prob = (new_prob / length_penalty) + coverage_penalty + eos_penalty
    return (new_seq, new_prob, eos)


def expand_kinya_outputs(model_setup, completed_outputs, input_outputs, logits, src_attn_weights,
                         max_morpho_inference_table_length,
                         max_batch_size, debug=False,
                         prob_cutoff=0.3, affix_prob_cutoff=0.3,
                         affix_min_prob=0.3, lprob_score_delta=2.0):
    global DECODE_TIME

    (en2kin_model, engl_roberta_model, kin_lm_model, ffi, lib, device) = model_setup

    (en_ids, outputs) = input_outputs
    (next_stems,
     next_pos_tags,
     next_lm_morphs,
     next_affixes, _) = logits

    ttt = time.time()
    entries = lib.klm_expand(ffi.cast("float *", next_stems.numpy(force=True).ctypes.data),
                             ffi.cast("float *", next_pos_tags.numpy(force=True).ctypes.data),
                             ffi.cast("float *", next_lm_morphs.numpy(force=True).ctypes.data),
                             ffi.cast("float *", next_affixes.numpy(force=True).ctypes.data),
                             next_stems.size(-1), next_pos_tags.size(-1), next_lm_morphs.size(-1), next_affixes.size(-1),
                             max_batch_size, max_morpho_inference_table_length,
                             prob_cutoff, affix_prob_cutoff, affix_min_prob, lprob_score_delta)
    DECODE_TIME += (time.time() - ttt)
    # src_attn_weights: N,T,S
    N = len(outputs)
    new_outputs = []
    for bidx in range(N):
        seq, log_prob, eos_flag = outputs[bidx]
        entry = entries[bidx]
        print('(entry != ffi.NULL):', (entry != ffi.NULL), flush=True)
        while (entry != ffi.NULL):
            (new_seq, new_prob, eos) = process_new_entry(bidx, seq, src_attn_weights, log_prob, entry, ffi)
            new_outputs.append((new_seq, new_prob, eos))
            entry = entry.next
    lib.klm_release_entries(entries, max_batch_size)

    new_completed_outputs = [x for x in new_outputs if x[2]]
    pending_outputs = [x for x in new_outputs if not x[2]]

    # Sort by output probability
    pending_outputs = sorted(pending_outputs, key=lambda x: x[1], reverse=True)
    pending_outputs = pending_outputs[:max_batch_size]

    completed_outputs = completed_outputs + new_completed_outputs
    completed_outputs = sorted(completed_outputs, key=lambda x: x[1], reverse=True)

    input_outputs = (en_ids, pending_outputs)
    return input_outputs, completed_outputs


def copy_batch_item_to_device(batch_data_item, device):
    (english_input_ids, english_sequence_lengths,
     lm_morphs, pos_tags, stems, input_sequence_lengths,
     afx_padded, m_masks_padded,
     affixes_prob, tokens_lengths,
     copy_tokens_prob,
     src_key_padding_mask, tgt_key_padding_mask, decoder_mask) = batch_data_item

    return (device, english_input_ids.to(device), english_sequence_lengths,
            lm_morphs.to(device), pos_tags.to(device), stems.to(device), input_sequence_lengths,
            afx_padded.to(device), m_masks_padded.to(device),
            affixes_prob.to(device) if (affixes_prob is not None) else None, tokens_lengths,
            (copy_tokens_prob.to(device) if (copy_tokens_prob is not None) else None),
            src_key_padding_mask.to(device), tgt_key_padding_mask.to(device), decoder_mask.to(device))

def read_english_lexicon(lexicon_file='english_lexicon.tsv'):
    lines = read_lines(lexicon_file)
    english_lexicon = dict()
    for line in lines:
        pieces = line.split('\t')
        if len(pieces) == 3:
            english_lexicon[pieces[0]] = (int(pieces[1]), int(pieces[2]))
    return english_lexicon

# def adjust_marks(txt):
#     if len(txt) > 1:
#         if txt[0] in '.?!?,;\"\'':
#             txt = txt[:1] + ' ' + adjust_marks(txt[1:])
#         if txt[-1] in '.?!?,;\"\'':
#             txt = adjust_marks(txt[:-1]) + ' ' + txt[-1:]
#     return txt

def adjust_english_sentence(english_sentence, english_lexicon):
    # english_sentence = adjust_marks(english_sentence.strip().rstrip().strip('\n').rstrip('\n').strip('\t').rstrip('\t'))
    english_sentence = ' '.join(english_sentence.replace('\t', ' ').replace('\n', ' ').split())
    if english_lexicon is not None:
        apostrophe_idx = english_sentence.index('\'') if ('\'' in english_sentence) else 0
        space_idx = english_sentence.index(' ') if (' ' in english_sentence) else 0
        idx = 0
        if (apostrophe_idx > 0) and (space_idx > 0):
            idx = min(apostrophe_idx, space_idx)
        elif apostrophe_idx > 0:
            idx = apostrophe_idx
        elif space_idx > 0:
            idx = space_idx
        if idx > 0:
            token = english_sentence[:idx]
            remaining = english_sentence[idx:]
        else:
            token = english_sentence
            remaining = ''
        key = token.lower()
        if key in english_lexicon:
            count,first_up = english_lexicon[key]
            ratio = float(first_up)/float(count)
            if ratio > 0.8:
                english_sentence = token[:1].upper()+token[1:]+remaining
            elif ratio < 0.2:
                english_sentence = token.lower()+remaining
    return english_sentence

def en2kin_translate(model_setup,
                     english_sentence,
                     max_text_length,
                     max_completed,
                     max_morpho_inference_table_length=20,
                     max_batch_size=8,
                     debug=False, ENGL_SEGMENT_PER_TOKEN = False,
                     english_lexicon=None,
                     prob_cutoff=0.3, affix_prob_cutoff=0.3,
                     affix_min_prob=0.3, lprob_score_delta=2.0):
    global GPU_TIME
    global TRANSLATE_TIME
    global TRANSLATION_SENTENCE_COUNT
    global TRANSLATION_TOKEN_COUNT
    global EXPANSION_TIME

    start_translate_time = time.time()

    (en2kin_model, engl_roberta_model, kin_lm_model, ffi, lib, device) = model_setup

    completed_outputs = []

    english_sentence = adjust_english_sentence(english_sentence, english_lexicon)

    input_outputs = initial_input_outputs(english_sentence, engl_roberta_model, ENGL_SEGMENT_PER_TOKEN = ENGL_SEGMENT_PER_TOKEN)

    TRANSLATION_SENTENCE_COUNT += 1.0
    TRANSLATION_TOKEN_COUNT += (1.0*len(input_outputs[0]))

    batch_data_item = batch_data(input_outputs, device)

    # Encoding
    with torch.no_grad():
        ttt = time.time()
        encoder_data = en2kin_model.encode_en(engl_roberta_model, batch_data_item)
        GPU_TIME += (time.time() - ttt)

    # Decoding
    en2kin_model.eval()
    engl_roberta_model.eval()
    seq_count = 0
    while True:
        with torch.no_grad():
            ttt = time.time()
            logits, src_attn_weights = en2kin_model.decode_kin(encoder_data, kin_lm_model, batch_data_item)
            GPU_TIME += (time.time() - ttt)

        qqq = time.time()
        input_outputs, completed_outputs = expand_kinya_outputs(model_setup,
                                                                completed_outputs, input_outputs,
                                                                logits, src_attn_weights,
                                                                max_morpho_inference_table_length, max_batch_size,
                                                                debug=debug,
                                                                prob_cutoff=prob_cutoff,
                                                                affix_prob_cutoff=affix_prob_cutoff,
                                                                affix_min_prob=affix_min_prob,
                                                                lprob_score_delta=lprob_score_delta)
        EXPANSION_TIME += (time.time() - qqq)

        seq_count += 1
        if (len(input_outputs[1]) <= 0) or (seq_count >= max_text_length) or (len(completed_outputs) >= max_completed):
            break
        batch_data_item = batch_data(input_outputs, device)
        if seq_count > 0:
            debug = False

    complete = sorted([(prob, decode_kinya_sequence(seq)) for seq, prob, eos in completed_outputs], key=lambda x: x[0],
                      reverse=True)
    pending = []
    if len(complete) == 0:
        pending = sorted([(prob, decode_kinya_sequence(seq)) for seq, prob, eos in input_outputs[1]],
                         key=lambda x: x[0],
                         reverse=True)
    TRANSLATE_TIME += (time.time() - start_translate_time)
    return complete, pending

def reset_and_print_en2kin_latency_info():
    global DECODE_TIME
    global GPU_TIME
    global TRANSLATE_TIME
    global EXPANSION_TIME
    global TRANSLATION_SENTENCE_COUNT
    global TRANSLATION_TOKEN_COUNT

    PRE_POST_PROC_TIME = TRANSLATE_TIME - (GPU_TIME + EXPANSION_TIME)
    BEAM_SEARCH_TIME = EXPANSION_TIME - DECODE_TIME
    if (TRANSLATION_SENTENCE_COUNT > 0):
        print('SENTENCE STATS:')
        print('Translation/sentence:   {:.3f} sec - {:.0f} %'.format(TRANSLATE_TIME/TRANSLATION_SENTENCE_COUNT, 100.0 * TRANSLATE_TIME / TRANSLATE_TIME))
        print('Morpho-Decode/sentence: {:.3f} sec - {:.0f} %'.format(DECODE_TIME/TRANSLATION_SENTENCE_COUNT, 100.0 * DECODE_TIME / TRANSLATE_TIME))
        print('GPU/sentence:           {:.3f} sec - {:.0f} %'.format(GPU_TIME/TRANSLATION_SENTENCE_COUNT, 100.0 * GPU_TIME / TRANSLATE_TIME))
        print('Expansion/sentence:     {:.3f} sec - {:.0f} %'.format(EXPANSION_TIME/TRANSLATION_SENTENCE_COUNT, 100.0 * EXPANSION_TIME / TRANSLATE_TIME))
        print('BeamSearch/sentence:    {:.3f} sec - {:.0f} %'.format(BEAM_SEARCH_TIME/TRANSLATION_SENTENCE_COUNT, 100.0 * BEAM_SEARCH_TIME / TRANSLATE_TIME))
        print('PrePostProc/sentence:   {:.3f} sec - {:.0f} %'.format(PRE_POST_PROC_TIME/TRANSLATION_SENTENCE_COUNT, 100.0 * PRE_POST_PROC_TIME / TRANSLATE_TIME))

        print('TOKEN STATS:')
        print('Translation/token:   {:.3f} msec - {:.0f} %'.format(1000.0 * TRANSLATE_TIME/TRANSLATION_TOKEN_COUNT, 100.0 * TRANSLATE_TIME / TRANSLATE_TIME))
        print('Morpho-Decode/token: {:.3f} msec - {:.0f} %'.format(1000.0 * DECODE_TIME/TRANSLATION_TOKEN_COUNT, 100.0 * DECODE_TIME / TRANSLATE_TIME))
        print('GPU/token:           {:.3f} msec - {:.0f} %'.format(1000.0 * GPU_TIME/TRANSLATION_TOKEN_COUNT, 100.0 * GPU_TIME / TRANSLATE_TIME))
        print('Expansion/token:     {:.3f} msec - {:.0f} %'.format(1000.0 * EXPANSION_TIME/TRANSLATION_TOKEN_COUNT, 100.0 * EXPANSION_TIME / TRANSLATE_TIME), flush=True)
        print('BeamSearch/token:    {:.3f} msec - {:.0f} %'.format(1000.0 * BEAM_SEARCH_TIME/TRANSLATION_TOKEN_COUNT, 100.0 * BEAM_SEARCH_TIME / TRANSLATE_TIME), flush=True)
        print('PrePostProc/token:   {:.3f} msec - {:.0f} %'.format(1000.0 * PRE_POST_PROC_TIME/TRANSLATION_TOKEN_COUNT, 100.0 * PRE_POST_PROC_TIME / TRANSLATE_TIME), flush=True)

    DECODE_TIME = 0.0
    GPU_TIME = 0.0
    TRANSLATE_TIME = 0.0
    TRANSLATION_SENTENCE_COUNT = 0.0
    TRANSLATION_TOKEN_COUNT = 0.0
    EXPANSION_TIME = 0.0

