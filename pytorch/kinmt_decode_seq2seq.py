import math
from itertools import accumulate
from typing import Tuple

import torch
import torch.nn.functional as F
from kinmt_data_seq2seq import generate_input_key_padding_mask, generate_square_subsequent_mask
from kinmt_seq2seq import Seq2SeqTransformer
from transformers import AutoTokenizer

def seq2seq_init_models(args, rank=0) -> Tuple[Seq2SeqTransformer,AutoTokenizer,AutoTokenizer, torch.device]:
    device = torch.device('cuda:%d' % rank)
    torch.cuda.set_device(rank)
    home_path = args.home_path
    curr_save_file_path = home_path + f"data/{args.kinmt_model_name}.pt"

    en_tokenizer = AutoTokenizer.from_pretrained('tokenizers/sentencepiece/english/')
    rw_tokenizer = AutoTokenizer.from_pretrained('tokenizers/sentencepiece/kinyarwanda/')
    if args.kinmt_seq2seq_config == 'en2kin':
        src_tokenizer, tgt_tokenizer = en_tokenizer, rw_tokenizer
    else:
        src_tokenizer, tgt_tokenizer = rw_tokenizer, en_tokenizer

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    model = Seq2SeqTransformer(args, use_cross_pos_attn=args.use_cross_positional_attn_bias).to(device)
    kb_state_dict = torch.load(curr_save_file_path, map_location=map_location)
    model.load_state_dict(kb_state_dict['model_state_dict'])

    return (model, src_tokenizer, tgt_tokenizer, device)

def initial_input_outputs(src_sentence, src_tokenizer, device):
    BOS_idx = 0
    EOS_idx = 2

    line_ids = []
    line_ids.append(('', [BOS_idx]))

    tokens_line = ' '.join(src_sentence.replace('<unk>', ' ').replace('<s>', ' ').replace('</s>', ' ').replace('\t', ' ').split())
    token_ids = src_tokenizer(tokens_line)['input_ids']
    if token_ids[0] != BOS_idx:
        token_ids = [BOS_idx] + token_ids
    if token_ids[-1] != EOS_idx:
        token_ids = token_ids + [EOS_idx]

    max_seq_len = 512
    if len(token_ids) > max_seq_len:
        token_ids = [token_ids[0]] + token_ids[1:(max_seq_len-1)] + [token_ids[-1]]

    src_input_ids = torch.tensor(token_ids).to(device)
    src_sequence_lengths = [len(token_ids)]

    tgt_input_ids = torch.tensor([BOS_idx], dtype=torch.long, device=device)
    tgt_sequence_lengths = [1]
    tgt_input_log_prob = [0.0]
    tgt_complete_sequences = []
    tgt_complete_sequences_prob = []

    seq_len = max(tgt_sequence_lengths)
    src_key_padding_mask = generate_input_key_padding_mask(src_sequence_lengths, ignore_last=False).to(device)
    tgt_key_padding_mask = generate_input_key_padding_mask(tgt_sequence_lengths, ignore_last=True).to(device)
    decoder_mask = generate_square_subsequent_mask(seq_len).to(device)

    batch_data_item = (device, src_input_ids, src_sequence_lengths,
                       tgt_input_ids, tgt_sequence_lengths,
                       src_key_padding_mask, tgt_key_padding_mask, decoder_mask)
    return (batch_data_item,
            tgt_input_log_prob,
            tgt_complete_sequences,
            tgt_complete_sequences_prob)

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


def seq2seq_beam_search_expand(tgt_input, tgt_input_lengths,
                              tgt_input_log_prob,
                              tgt_top_logits, tgt_top_idx,  # (N,beam_size)
                              tgt_EOS,
                              device, max_batch_size,
                              tgt_complete_sequences, tgt_complete_sequences_prob,
                              src_attn_weights,
                              max_seq_length=512):
    next_tgt_inputs = []
    next_tgt_inputs_prob = []
    # src_attn_weights: N,T,S
    tgt_input_boundaries = [(x - y, x) for x, y in zip(accumulate(tgt_input_lengths), tgt_input_lengths)]
    for n in range(tgt_top_logits.size(0)):  # n: batch
        (inp_start, inp_end) = tgt_input_boundaries[n]
        lprob = F.log_softmax(tgt_top_logits[n, :],
                              dim=-1).cpu().detach().numpy()  # shape: (beam_size), normalizing by top BS outputs
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

def decode_tgt(tgt_tokenizer, seq, SOS_idx = 0, EOS_idx = 2) -> str:
    if len(seq) == 0:
        return ''
    if seq[0] == SOS_idx:
        seq = seq[1:]
    if len(seq) == 0:
        return ''
    if seq[-1] == EOS_idx:
        seq = seq[:-1]
    if len(seq) == 0:
        return ''
    return tgt_tokenizer.decode(seq)

def seq2seq_translate(model_setup: Tuple[Seq2SeqTransformer,AutoTokenizer,AutoTokenizer, torch.device],
                     src_sentence,
                     beam_size, max_eval_sequences, max_text_length):
    tgt_EOS = 2
    (model, src_tokenizer, tgt_tokenizer, device) = model_setup
    model.eval()
    (batch_data_item,
     tgt_input_log_prob,
     tgt_complete_sequences,
     tgt_complete_sequences_prob) = initial_input_outputs(src_sentence, src_tokenizer, device)

    with torch.no_grad():
        (enc_output,
         enc_attn_bias,
         bert_src,
         src_key_padding_mask) = model.encode_src(batch_data_item)

    (device, src_input_ids, src_sequence_lengths,
     tgt_input, tgt_input_lengths,
     src_key_padding_mask, tgt_key_padding_mask, decoder_mask) = batch_data_item
    total_length = 0
    while True:
        with torch.no_grad():
            (tgt_out, _), src_attn_weights = model.decode_tgt(enc_output, enc_attn_bias, bert_src,
                                                              src_key_padding_mask,
                                                              tgt_input, tgt_input_lengths)
        tgt_top_logits, tgt_top_idx = tgt_out.topk(beam_size, dim=-1)  # (N,beam_size)
        (tgt_input, tgt_input_lengths,
         tgt_input_log_prob,
         tgt_complete_sequences,
         tgt_complete_sequences_prob) = seq2seq_beam_search_expand(tgt_input, tgt_input_lengths, tgt_input_log_prob,
                                                                  tgt_top_logits, tgt_top_idx,  # (L,N,beam_size)
                                                                  tgt_EOS,
                                                                  device, beam_size,
                                                                  tgt_complete_sequences, tgt_complete_sequences_prob,
                                                                  src_attn_weights)
        if (len(tgt_complete_sequences) >= max_eval_sequences) or (tgt_input is None):
            break
        if (total_length > max_text_length):
            break
        total_length += 1
    complete_translations = []
    for prob, seq in zip(tgt_complete_sequences_prob, tgt_complete_sequences):
        complete_translations.append((prob, decode_tgt(tgt_tokenizer, seq)))
    complete_translations = sorted(complete_translations, reverse=True, key=lambda x: x[0])

    pending_translations = []
    if len(complete_translations) == 0:
        start = 0
        for prob, tgt_len in zip(tgt_input_log_prob, tgt_input_lengths):
            end = start + tgt_len
            pending_translations.append((prob, decode_tgt(tgt_tokenizer,tgt_input[start:end]))) # Skip <sos>
            start = end
        pending_translations = sorted(pending_translations, reverse=True, key=lambda x: x[0])

    return complete_translations, pending_translations

