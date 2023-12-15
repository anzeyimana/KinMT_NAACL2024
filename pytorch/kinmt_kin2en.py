from typing import Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from kb_transformers import KINMT_TransformerEncoderLayer, KINMT_TransformerDecoderLayer, KINMT_Transformer, init_bert_params
from kinmt_data import generate_input_key_padding_mask, generate_square_subsequent_mask
from mybert import MyBERTEncoder
from modules import PositionEncoding, MorphoEncoder, BaseConfig, Engl_TokenGenerator, KINMT_HeadTransform
from fairseq.models.transformer_lm import TransformerLanguageModel

EN_PAD_IDX = 1
KIN_PAD_IDX = 0

def mybert_input(mybert_encoder: MyBERTEncoder, lm_morphs, pos_tags, stems, input_sequence_lengths,
                    afx_padded, m_masks_padded, src_key_padding_mask):
    bert_src = mybert_encoder(lm_morphs, pos_tags, stems, input_sequence_lengths, afx_padded, m_masks_padded, src_key_padding_mask)
    return bert_src

def english_lm_input(english_lm: TransformerLanguageModel, english_input_ids, english_sequence_lengths, input_with_eos = True):
    lists = english_input_ids.split(english_sequence_lengths, 0)
    tr_padded = pad_sequence(lists, batch_first=True, padding_value=EN_PAD_IDX)  # (N,L)
    if input_with_eos:
        for i,l in enumerate(english_sequence_lengths):
            tr_padded[i,(l-1)] = 1
    lm_tgt, _ = english_lm.extract_features(tr_padded, encoder_out=None) # N x L --> N x L x E
    lm_tgt = lm_tgt.transpose(0, 1)  # Shape: L x N x E, with L = max sequence length
    return lm_tgt

class Kin2EnTransformer(nn.Module):
    def __init__(self, args, cfg:BaseConfig,
                 mybert_encoder: Union[MyBERTEncoder,None], english_lm: Union[TransformerLanguageModel,None],
                 use_cross_pos_attn: bool = True):
        super(Kin2EnTransformer, self).__init__()
        self.hidden_dim = (args.morpho_dim_hidden * 4) + args.stem_dim_hidden # 128 x 4 + 256 = 768
        self.num_heads = args.main_sequence_encoder_num_heads
        self.en_vocab_size = english_lm.decoder.embed_tokens.weight.shape[0] if english_lm is not None else 42022
        self.en_embed_size = english_lm.decoder.embed_tokens.weight.shape[1] if english_lm is not None else 1024

        if mybert_encoder is not None:
            for param in mybert_encoder.parameters():
                param.requires_grad = False

        if english_lm is not None:
            for param in english_lm.parameters():
                param.requires_grad = False

        args.morpho_num_layers = 3

        self.tgt_token_embedding = nn.Embedding(self.en_vocab_size, self.hidden_dim, padding_idx=EN_PAD_IDX)
        self.tgt_pos_encoder = PositionEncoding(self.hidden_dim,
                                            self.num_heads,
                                            args.main_sequence_encoder_max_seq_len,
                                            args.main_sequence_encoder_rel_pos_bins,
                                            args.main_sequence_encoder_max_rel_pos,
                                            False)
        if mybert_encoder is not None:
            self.src_bert_proj = KINMT_HeadTransform(mybert_encoder.hidden_dim, self.hidden_dim,
                                                     args.layernorm_epsilon, dropout=0.3)

        if english_lm is not None:
            self.tgt_lm_proj = KINMT_HeadTransform(self.en_embed_size, self.hidden_dim,
                                                   args.layernorm_epsilon, dropout=0.3)

        self.src_morpho_encoder = MorphoEncoder(args,cfg)
        self.src_stem_embedding = nn.Embedding(cfg.tot_num_stems, args.stem_dim_hidden, padding_idx=KIN_PAD_IDX)
        self.src_pos_encoder = PositionEncoding(self.hidden_dim,
                                            self.num_heads,
                                            args.main_sequence_encoder_max_seq_len,
                                            args.main_sequence_encoder_rel_pos_bins,
                                            args.main_sequence_encoder_max_rel_pos,
                                            False)

        self.generator = Engl_TokenGenerator(self.tgt_token_embedding.weight, self.hidden_dim, args.layernorm_epsilon,
                                             copy_tokens_embedding_weights=(self.src_stem_embedding.weight if args.kinmt_use_copy_loss else None))

        encoder_layer = KINMT_TransformerEncoderLayer(self.hidden_dim, args.main_sequence_encoder_num_heads,
                                                      dim_feedforward=args.main_sequence_encoder_dim_ffn,
                                                      dropout=0.3, activation="relu",
                                                      bert=(mybert_encoder is not None))
        decoder_layer = KINMT_TransformerDecoderLayer(self.hidden_dim, args.main_sequence_encoder_num_heads,
                                                      dim_feedforward=args.main_sequence_encoder_dim_ffn,
                                                      dropout=0.3, activation="relu",
                                                      bert=(mybert_encoder is not None),
                                                      gpt=(english_lm is not None))
        self.transformer = KINMT_Transformer(encoder_layer, decoder_layer,
                                             8 if (mybert_encoder is None) else 5, 8 if (english_lm is None) else 7,
                                             bert=(mybert_encoder is not None),
                                             gpt=(english_lm is not None),
                                             use_cross_pos_attn = use_cross_pos_attn)
        self.apply(init_bert_params)

    def tgt_english_input(self, english_input_ids, english_sequence_lengths):
        input_sequences = self.tgt_token_embedding(english_input_ids)  # (L, E')
        lists = input_sequences.split(english_sequence_lengths, 0) # len(input_sequence_lengths) = N (i.e. Batch Size, e.g. 32)
        dec_input = pad_sequence(lists, batch_first=False)  # (L,N,E)
        dec_attn_bias = self.tgt_pos_encoder(dec_input)
        return dec_input, dec_attn_bias

    def src_kinya_input(self, lm_morphs, pos_tags, stems, input_sequence_lengths,
                        afx_padded, m_masks_padded):
        morpho_input = self.src_morpho_encoder(stems, lm_morphs, pos_tags, afx_padded, m_masks_padded)  # [4, L, E1]
        stem_input = self.src_stem_embedding(stems)  # [L, E2]
        morpho_input = morpho_input.permute(1, 0, 2)  # ==> [L, 4, E1]
        L = morpho_input.size(0)
        morpho_input = morpho_input.contiguous().view(L, -1)  # (L, 4E1)
        input_sequences = torch.cat((morpho_input, stem_input), 1)  # [L, E'=4E1+E2]
        lists = input_sequences.split(input_sequence_lengths, 0)  # len(input_sequence_lengths)
        enc_input = pad_sequence(lists, batch_first=False)
        enc_attn_bias = self.src_pos_encoder(enc_input)
        return enc_input, enc_attn_bias

    def process_inputs(self, mybert_encoder: MyBERTEncoder, english_lm: TransformerLanguageModel,
                       english_input_ids, english_sequence_lengths,
                       lm_morphs, pos_tags, stems, input_sequence_lengths,
                       afx_padded, m_masks_padded,
                       src_key_padding_mask, tgt_key_padding_mask, decoder_mask):
        if mybert_encoder is not None:
            with torch.no_grad():
                bert_src = mybert_input(mybert_encoder, lm_morphs, pos_tags, stems, input_sequence_lengths,
                                afx_padded, m_masks_padded, src_key_padding_mask)
            # This part is trainable
            bert_src = self.src_bert_proj(bert_src)
            # torch.cuda.empty_cache()
        else:
            bert_src = None

        if english_lm is not None:
            with torch.no_grad():
                lm_tgt = english_lm_input(english_lm, english_input_ids, english_sequence_lengths,
                                          input_with_eos = True)
            # This part is trainable
            lm_tgt = self.tgt_lm_proj(lm_tgt)
            # torch.cuda.empty_cache()
        else:
            lm_tgt = None

        enc_input, enc_attn_bias = self.src_kinya_input(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                                        afx_padded, m_masks_padded)
        dec_input, dec_attn_bias = self.tgt_english_input(english_input_ids, english_sequence_lengths)

        transformer_out = self.transformer(enc_input, dec_input,
                                           src_attn_bias = enc_attn_bias, src_key_padding_mask = src_key_padding_mask,
                                           tgt_mask = decoder_mask, tgt_attn_bias = dec_attn_bias, tgt_key_padding_mask = tgt_key_padding_mask,
                                           src_bert = bert_src,
                                           tgt_gpt = lm_tgt)
        return transformer_out

    def forward(self, mybert_encoder: MyBERTEncoder, english_lm: TransformerLanguageModel, batch_data_item):
        (device, english_input_ids, english_sequence_lengths,
         lm_morphs, pos_tags, stems, input_sequence_lengths,
         afx_padded, m_masks_padded,
         affixes_prob, tokens_lengths,
         copy_tokens_prob,
         src_key_padding_mask, tgt_key_padding_mask, decoder_mask) = batch_data_item
        transformer_out = self.process_inputs(mybert_encoder, english_lm,
                                              english_input_ids, english_sequence_lengths,
                                              lm_morphs, pos_tags, stems, input_sequence_lengths,
                                              afx_padded, m_masks_padded,
                                              src_key_padding_mask, tgt_key_padding_mask, decoder_mask)
        return self.generator(transformer_out, english_input_ids, english_sequence_lengths,
                              copy_tokens_prob = copy_tokens_prob)

    def encode_kin(self, mybert_encoder: MyBERTEncoder, seed_data_item):
        (lm_morphs, pos_tags, stems, input_sequence_lengths,
         afx_padded, m_masks_padded,
         affixes_prob, tokens_lengths,
         src_key_padding_mask) = seed_data_item

        # Process for Kinya side
        if mybert_encoder is not None:
            with torch.no_grad():
                bert_src = mybert_input(mybert_encoder, lm_morphs, pos_tags, stems, input_sequence_lengths,
                                afx_padded, m_masks_padded, src_key_padding_mask)
            # This part is trainable
            bert_src = self.src_bert_proj(bert_src)
            # torch.cuda.empty_cache()
        else:
            bert_src = None

        enc_input, enc_attn_bias = self.src_kinya_input(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                                        afx_padded, m_masks_padded)

        enc_output = self.transformer.encode(enc_input,
                                             src_attn_bias = enc_attn_bias, src_key_padding_mask = src_key_padding_mask,
                                             src_bert = bert_src)
        return enc_output, enc_attn_bias, bert_src, src_key_padding_mask

    def decode_en(self, english_lm: TransformerLanguageModel,
                  enc_output, enc_attn_bias, bert_src, src_key_padding_mask,
                  english_input_ids, english_sequence_lengths):
        seq_len = max(english_sequence_lengths)
        tgt_key_padding_mask = generate_input_key_padding_mask(english_sequence_lengths, ignore_last=False).to(enc_output.device)
        #decoder_mask = None# No masking output context needed because we are decoding# generate_square_subsequent_mask(seq_len, enc_output.device)
        decoder_mask = generate_square_subsequent_mask(seq_len).to(enc_output.device)

        # Process English data
        if english_lm is not None:
            with torch.no_grad():
                lm_tgt = english_lm_input(english_lm, english_input_ids, english_sequence_lengths,
                                          input_with_eos = False)
            # This part is trainable
            lm_tgt = self.tgt_lm_proj(lm_tgt)
            # torch.cuda.empty_cache()
        else:
            lm_tgt = None

        dec_input, dec_attn_bias = self.tgt_english_input(english_input_ids, english_sequence_lengths)
        N = len(english_sequence_lengths)
        transformer_out, src_attn_weights = self.transformer.decode(enc_output.repeat(1,N,1), dec_input,
                                           src_key_padding_mask = src_key_padding_mask.repeat(N,1),
                                           tgt_mask = decoder_mask, tgt_attn_bias = dec_attn_bias, tgt_key_padding_mask = tgt_key_padding_mask,
                                           src_bert = bert_src.repeat(1,N,1) if bert_src is not None else None,
                                           tgt_gpt = lm_tgt)
        (token_scores, next_copy_prob) = self.generator.predict(transformer_out, english_sequence_lengths)
        return (token_scores, next_copy_prob), src_attn_weights
