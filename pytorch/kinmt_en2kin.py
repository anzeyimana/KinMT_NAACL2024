from typing import Union

import torch
import torch.nn as nn
from kb_transformers import KINMT_TransformerEncoderLayer, KINMT_TransformerDecoderLayer, \
    KINMT_Transformer, init_bert_params
from mygpt import MyGPTEncoder
from modules import PositionEncoding, MorphoEncoder, Kinya_TokenGenerator, BaseConfig, KINMT_HeadTransform
from fairseq.models.roberta import RobertaModel
from torch.nn.utils.rnn import pad_sequence

EN_PAD_IDX = 1
KIN_PAD_IDX = 0

def kin_lm_input(kin_lm_model: MyGPTEncoder, lm_morphs, pos_tags, stems, input_sequence_lengths,
                 afx_padded, m_masks_padded, tgt_key_padding_mask, decoder_mask):
    lm_tgt = kin_lm_model(lm_morphs, pos_tags, stems, input_sequence_lengths,
                       afx_padded, m_masks_padded, tgt_key_padding_mask, decoder_mask)
    return lm_tgt

def bert_input(roberta_model: RobertaModel, english_input_ids, english_sequence_lengths):
    lists = english_input_ids.split(english_sequence_lengths, 0)  # len(input_sequence_lengths) = N (i.e. Batch Size, e.g. 32)
    tr_padded = pad_sequence(lists, batch_first=True, padding_value=EN_PAD_IDX)  # (N,L) # English pad=1
    bert_src = roberta_model.extract_features(tr_padded) # N x L --> N x L x E
    bert_src = bert_src.transpose(0, 1)  # Shape: L x N x E, with L = max sequence length
    return bert_src

class En2KinTransformer(nn.Module):
    def __init__(self, args, cfg:BaseConfig,
                 bert_encoder: Union[RobertaModel,None], my_gpt: Union[MyGPTEncoder,None],
                 use_cross_pos_attn: bool = True,
                 use_balanced_morpho: bool = False):
        super(En2KinTransformer, self).__init__()
        self.use_balanced_morpho = use_balanced_morpho
        if self.use_balanced_morpho:
            args.morpho_dim_hidden = 240
            self.hidden_dim = (args.morpho_dim_hidden * 4)
        else:
            self.hidden_dim = (args.morpho_dim_hidden * 4) + args.stem_dim_hidden # 128 x 4 + 256 = 768
        self.num_heads = args.main_sequence_encoder_num_heads
        self.en_vocab_size = bert_encoder.model.encoder.sentence_encoder.embed_tokens.weight.shape[0] if bert_encoder is not None else 50265
        self.en_embed_size = bert_encoder.model.encoder.sentence_encoder.embed_tokens.weight.shape[1] if bert_encoder is not None else 768

        if bert_encoder is not None:
            for param in bert_encoder.parameters():
                param.requires_grad = False

        if my_gpt is not None:
            for param in my_gpt.parameters():
                param.requires_grad = False

        args.morpho_num_layers = 3

        self.src_token_embedding = nn.Embedding(self.en_vocab_size, self.hidden_dim, padding_idx=EN_PAD_IDX) # English PAD=1
        self.src_pos_encoder = PositionEncoding(self.hidden_dim,
                                            self.num_heads,
                                            args.main_sequence_encoder_max_seq_len,
                                            args.main_sequence_encoder_rel_pos_bins,
                                            args.main_sequence_encoder_max_rel_pos,
                                            False)
        if bert_encoder is not None:
            self.src_bert_proj = KINMT_HeadTransform(self.en_embed_size, self.hidden_dim,
                                                     args.layernorm_epsilon, dropout=0.3)

        if my_gpt is not None:
            self.tgt_lm_proj = KINMT_HeadTransform(my_gpt.hidden_dim, self.hidden_dim,
                                                   args.layernorm_epsilon, dropout=0.3)

        self.tgt_morpho_encoder = MorphoEncoder(args,cfg)
        self.tgt_pos_encoder = PositionEncoding(self.hidden_dim,
                                            self.num_heads,
                                            args.main_sequence_encoder_max_seq_len,
                                            args.main_sequence_encoder_rel_pos_bins,
                                            args.main_sequence_encoder_max_rel_pos,
                                            False)
        if not self.use_balanced_morpho:
            self.tgt_stem_embedding = nn.Embedding(cfg.tot_num_stems, args.stem_dim_hidden, padding_idx=KIN_PAD_IDX)
            self.generator = Kinya_TokenGenerator(self.tgt_stem_embedding.weight,
                                                  self.tgt_morpho_encoder.pos_tag_embedding.weight,
                                                  self.tgt_morpho_encoder.lm_morph_one_embedding.weight,
                                                  self.tgt_morpho_encoder.affixes_embedding.weight,
                                                  self.hidden_dim,
                                                  args.layernorm_epsilon,
                                                  copy_tokens_embedding_weights=(self.src_token_embedding.weight if args.kinmt_use_copy_loss else None))
        else:
            self.generator = Kinya_TokenGenerator(self.tgt_morpho_encoder.stem_embedding.weight,
                                                  self.tgt_morpho_encoder.pos_tag_embedding.weight,
                                                  self.tgt_morpho_encoder.lm_morph_one_embedding.weight,
                                                  self.tgt_morpho_encoder.affixes_embedding.weight,
                                                  self.hidden_dim,
                                                  args.layernorm_epsilon,
                                                  copy_tokens_embedding_weights=(self.src_token_embedding.weight if args.kinmt_use_copy_loss else None))

        encoder_layer = KINMT_TransformerEncoderLayer(self.hidden_dim, args.main_sequence_encoder_num_heads,
                                                      dim_feedforward=args.main_sequence_encoder_dim_ffn,
                                                      dropout=0.3, activation="relu",
                                                      bert=(bert_encoder is not None))
        decoder_layer = KINMT_TransformerDecoderLayer(self.hidden_dim, args.main_sequence_encoder_num_heads,
                                                      dim_feedforward=args.main_sequence_encoder_dim_ffn,
                                                      dropout=0.3, activation="relu",
                                                      bert=(bert_encoder is not None),
                                                      gpt=(my_gpt is not None))
        self.transformer = KINMT_Transformer(encoder_layer, decoder_layer,
                                             8 if (bert_encoder is None) else 5, 8 if (my_gpt is None) else 7,
                                             bert=(bert_encoder is not None),
                                             gpt=(my_gpt is not None),
                                             use_cross_pos_attn = use_cross_pos_attn)
        self.apply(init_bert_params)

    def src_english_input(self, english_input_ids, english_sequence_lengths):
        input_sequences = self.src_token_embedding(english_input_ids)  # (L, E')
        lists = input_sequences.split(english_sequence_lengths, 0) # len(input_sequence_lengths) = N (i.e. Batch Size, e.g. 32)
        enc_input = pad_sequence(lists, batch_first=False)  # (L,N,E)
        enc_attn_bias = self.src_pos_encoder(enc_input)
        return enc_input, enc_attn_bias

    def tgt_kinya_input(self, lm_morphs, pos_tags, stems, input_sequence_lengths,
                        afx_padded, m_masks_padded):
        morpho_input = self.tgt_morpho_encoder(stems, lm_morphs, pos_tags, afx_padded, m_masks_padded)  # [4, L, E1]
        morpho_input = morpho_input.permute(1, 0, 2)  # ==> [L, 4, E1]
        L = morpho_input.size(0)
        morpho_input = morpho_input.contiguous().view(L, -1)  # (L, 4E1)

        if not self.use_balanced_morpho:
            stem_input = self.tgt_stem_embedding(stems)  # [L, E2]
            morpho_input = torch.cat((morpho_input, stem_input), 1)  # [L, E'=4E1+E2]

        lists = morpho_input.split(input_sequence_lengths, 0)  # len(input_sequence_lengths)
        dec_input = pad_sequence(lists, batch_first=False)
        dec_attn_bias = self.tgt_pos_encoder(dec_input)
        return dec_input, dec_attn_bias

    def process_inputs(self, roberta_model : RobertaModel, kin_lm_model: MyGPTEncoder,
                       english_input_ids, english_sequence_lengths,
                       lm_morphs, pos_tags, stems, input_sequence_lengths,
                       afx_padded, m_masks_padded,
                       src_key_padding_mask, tgt_key_padding_mask, decoder_mask):
        if roberta_model is not None:
            with torch.no_grad():
                bert_src = bert_input(roberta_model, english_input_ids, english_sequence_lengths)
            # This part is trainable
            bert_src = self.src_bert_proj(bert_src)
            # torch.cuda.empty_cache()
        else:
            bert_src = None

        if kin_lm_model is not None:
            with torch.no_grad():
                lm_tgt = kin_lm_input(kin_lm_model, lm_morphs, pos_tags, stems, input_sequence_lengths,
                                      afx_padded, m_masks_padded, tgt_key_padding_mask, decoder_mask)
            # This part is trainable
            lm_tgt = self.tgt_lm_proj(lm_tgt)
            # torch.cuda.empty_cache()
        else:
            lm_tgt = None

        enc_input, enc_attn_bias = self.src_english_input(english_input_ids, english_sequence_lengths)
        dec_input, dec_attn_bias = self.tgt_kinya_input(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                                        afx_padded, m_masks_padded)

        transformer_out = self.transformer(enc_input, dec_input,
                                           src_attn_bias = enc_attn_bias, src_key_padding_mask = src_key_padding_mask,
                                           tgt_mask = decoder_mask, tgt_attn_bias = dec_attn_bias, tgt_key_padding_mask = tgt_key_padding_mask,
                                           src_bert = bert_src,
                                           tgt_gpt = lm_tgt)
        return transformer_out

    def forward(self, roberta_model: RobertaModel, kin_lm_model: MyGPTEncoder, batch_data_item):
        (device, english_input_ids, english_sequence_lengths,
         lm_morphs, pos_tags, stems, input_sequence_lengths,
         afx_padded, m_masks_padded,
         affixes_prob, tokens_lengths,
         copy_tokens_prob,
         src_key_padding_mask, tgt_key_padding_mask, decoder_mask) = batch_data_item
        transformer_out = self.process_inputs(roberta_model, kin_lm_model,
                                              english_input_ids, english_sequence_lengths,
                                              lm_morphs, pos_tags, stems, input_sequence_lengths,
                                              afx_padded, m_masks_padded,
                                              src_key_padding_mask, tgt_key_padding_mask, decoder_mask)
        return self.generator(transformer_out, input_sequence_lengths, stems, pos_tags, lm_morphs,
                              affixes_prob, copy_tokens_prob = copy_tokens_prob)
    def encode_en(self, roberta_model: RobertaModel, batch_data_item):
        (device, english_input_ids, english_sequence_lengths,
         lm_morphs, pos_tags, stems, input_sequence_lengths,
         afx_padded, m_masks_padded,
         affixes_prob, tokens_lengths,
         src_key_padding_mask, tgt_key_padding_mask, decoder_mask) = batch_data_item

        if roberta_model is not None:
            with torch.no_grad():
                bert_src = bert_input(roberta_model, english_input_ids, english_sequence_lengths)
            # This part is trainable
            bert_src = self.src_bert_proj(bert_src)
            # torch.cuda.empty_cache()
        else:
            bert_src = None

        enc_input, enc_attn_bias = self.src_english_input(english_input_ids, english_sequence_lengths)

        encoder_output = self.transformer.encode(enc_input,
                                             src_attn_bias = enc_attn_bias, src_key_padding_mask = src_key_padding_mask,
                                             src_bert = bert_src)
        return encoder_output, enc_attn_bias, bert_src, src_key_padding_mask

    def decode_kin(self, encoder_data, kin_lm_model: MyGPTEncoder, batch_data_item):
        (encoder_output, enc_attn_bias, bert_src, src_key_padding_mask) = encoder_data
        (device, english_input_ids, english_sequence_lengths,
         lm_morphs, pos_tags, stems, input_sequence_lengths,
         afx_padded, m_masks_padded,
         affixes_prob, tokens_lengths,
         src_key_padding_mask, tgt_key_padding_mask, decoder_mask) = batch_data_item

        if kin_lm_model is not None:
            with torch.no_grad():
                lm_tgt = kin_lm_input(kin_lm_model, lm_morphs, pos_tags, stems, input_sequence_lengths,
                                      afx_padded, m_masks_padded, tgt_key_padding_mask, decoder_mask)
            # This part is trainable
            lm_tgt = self.tgt_lm_proj(lm_tgt)
            # torch.cuda.empty_cache()
        else:
            lm_tgt = None

        dec_input, dec_attn_bias = self.tgt_kinya_input(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                                        afx_padded, m_masks_padded)
        N = dec_input.size(1)
        transformer_out, src_attn_weights = self.transformer.decode(encoder_output.repeat(1, N, 1), dec_input,
                                                                    src_key_padding_mask=src_key_padding_mask.repeat(N, 1),
                                                                    tgt_mask=decoder_mask, tgt_attn_bias=dec_attn_bias,
                                                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                                                    src_bert=bert_src.repeat(1, N, 1) if bert_src is not None else None,
                                                                    tgt_gpt=lm_tgt)
        (next_stems, next_pos_tags, next_lm_morphs, next_affixes, next_copy_tokens) = self.generator.predict(transformer_out, input_sequence_lengths)
        return (next_stems, next_pos_tags, next_lm_morphs, next_affixes, next_copy_tokens), src_attn_weights
