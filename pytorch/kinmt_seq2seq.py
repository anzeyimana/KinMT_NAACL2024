import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from kb_transformers import KINMT_TransformerEncoderLayer, KINMT_TransformerDecoderLayer, \
    KINMT_Transformer, init_bert_params
from kinmt_data import generate_input_key_padding_mask, generate_square_subsequent_mask
from modules import PositionEncoding, Engl_TokenGenerator

SRC_PAD_IDX = 1
TGT_PAD_IDX = 1

class Seq2SeqTransformer(nn.Module):
    def __init__(self, args,
                 src_vocab_size=32_000, tgt_vocab_size=32_000,
                 src_pad_idx=1, tgt_pad_idx=1,
                 use_cross_pos_attn: bool = True,):
        super(Seq2SeqTransformer, self).__init__()
        self.hidden_dim = 768
        self.num_heads = args.main_sequence_encoder_num_heads
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.src_token_embedding = nn.Embedding(self.src_vocab_size, self.hidden_dim, padding_idx=src_pad_idx)
        self.src_pos_encoder = PositionEncoding(self.hidden_dim,
                                            self.num_heads,
                                            args.main_sequence_encoder_max_seq_len,
                                            args.main_sequence_encoder_rel_pos_bins,
                                            args.main_sequence_encoder_max_rel_pos,
                                            False)

        self.tgt_token_embedding = nn.Embedding(self.tgt_vocab_size, self.hidden_dim, padding_idx=tgt_pad_idx)
        self.tgt_pos_encoder = PositionEncoding(self.hidden_dim,
                                            self.num_heads,
                                            args.main_sequence_encoder_max_seq_len,
                                            args.main_sequence_encoder_rel_pos_bins,
                                            args.main_sequence_encoder_max_rel_pos,
                                            False)

        self.generator = Engl_TokenGenerator(self.tgt_token_embedding.weight, self.hidden_dim, args.layernorm_epsilon)

        encoder_layer = KINMT_TransformerEncoderLayer(self.hidden_dim, args.main_sequence_encoder_num_heads,
                                                      dim_feedforward=args.main_sequence_encoder_dim_ffn,
                                                      dropout=0.3, activation="relu",
                                                      bert=False)
        decoder_layer = KINMT_TransformerDecoderLayer(self.hidden_dim, args.main_sequence_encoder_num_heads,
                                                      dim_feedforward=args.main_sequence_encoder_dim_ffn,
                                                      dropout=0.3, activation="relu",
                                                      bert=False,
                                                      gpt=False)
        self.transformer = KINMT_Transformer(encoder_layer, decoder_layer, 8, 8,
                                             bert=False,
                                             gpt=False,
                                             use_cross_pos_attn = use_cross_pos_attn)
        self.apply(init_bert_params)

    def src_input(self, src_input_ids, src_sequence_lengths):
        input_sequences = self.src_token_embedding(src_input_ids)  # (L, E')
        lists = input_sequences.split(src_sequence_lengths, 0) # len(input_sequence_lengths) = N (i.e. Batch Size, e.g. 32)
        enc_input = pad_sequence(lists, batch_first=False)  # (L,N,E)
        enc_attn_bias = self.src_pos_encoder(enc_input)
        return enc_input, enc_attn_bias

    def tgt_input(self, tgt_input_ids, tgt_sequence_lengths):
        input_sequences = self.tgt_token_embedding(tgt_input_ids)  # (L, E')
        lists = input_sequences.split(tgt_sequence_lengths, 0) # len(input_sequence_lengths) = N (i.e. Batch Size, e.g. 32)
        dec_input = pad_sequence(lists, batch_first=False)  # (L,N,E)
        dec_attn_bias = self.tgt_pos_encoder(dec_input)
        return dec_input, dec_attn_bias

    def process_inputs(self, src_input_ids, src_sequence_lengths,
                       tgt_input_ids, tgt_sequence_lengths,
                       src_key_padding_mask, tgt_key_padding_mask, decoder_mask):
        bert_src = None
        lm_tgt = None
        enc_input, enc_attn_bias = self.src_input(src_input_ids, src_sequence_lengths)
        dec_input, dec_attn_bias = self.tgt_input(tgt_input_ids, tgt_sequence_lengths)

        transformer_out = self.transformer(enc_input, dec_input,
                                           src_attn_bias = enc_attn_bias, src_key_padding_mask = src_key_padding_mask,
                                           tgt_mask = decoder_mask, tgt_attn_bias = dec_attn_bias, tgt_key_padding_mask = tgt_key_padding_mask,
                                           src_bert = bert_src,
                                           tgt_gpt = lm_tgt)
        return transformer_out

    def forward(self, batch_data_item):
        (device, src_input_ids, src_sequence_lengths,
         tgt_input_ids, tgt_sequence_lengths,
         src_key_padding_mask, tgt_key_padding_mask, decoder_mask) = batch_data_item
        transformer_out = self.process_inputs(src_input_ids, src_sequence_lengths,
                                              tgt_input_ids, tgt_sequence_lengths,
                                              src_key_padding_mask, tgt_key_padding_mask, decoder_mask)
        return self.generator(transformer_out, tgt_input_ids, tgt_sequence_lengths)

    def encode_src(self, batch_data_item):
        (device, src_input_ids, src_sequence_lengths,
         tgt_input_ids, tgt_sequence_lengths,
         src_key_padding_mask, tgt_key_padding_mask, decoder_mask) = batch_data_item
        bert_src = None
        enc_input, enc_attn_bias = self.src_input(src_input_ids, src_sequence_lengths)
        encoder_output = self.transformer.encode(enc_input,
                                             src_attn_bias = enc_attn_bias, src_key_padding_mask = src_key_padding_mask,
                                             src_bert = bert_src)
        return encoder_output, enc_attn_bias, bert_src, src_key_padding_mask


    def decode_tgt(self, enc_output, enc_attn_bias, bert_src,
                  src_key_padding_mask,
                  tgt_input_ids, tgt_sequence_lengths):
        seq_len = max(tgt_sequence_lengths)
        tgt_key_padding_mask = generate_input_key_padding_mask(tgt_sequence_lengths, ignore_last=False).to(enc_output.device)
        #decoder_mask = None# No masking output context needed because we are decoding# generate_square_subsequent_mask(seq_len, enc_output.device)
        decoder_mask = generate_square_subsequent_mask(seq_len).to(enc_output.device)
        lm_tgt = None

        dec_input, dec_attn_bias = self.tgt_input(tgt_input_ids, tgt_sequence_lengths)
        N = len(tgt_sequence_lengths)
        transformer_out, src_attn_weights = self.transformer.decode(enc_output.repeat(1,N,1), dec_input,
                                           src_key_padding_mask = src_key_padding_mask.repeat(N,1),
                                           tgt_mask = decoder_mask, tgt_attn_bias = dec_attn_bias, tgt_key_padding_mask = tgt_key_padding_mask,
                                           src_bert = bert_src.repeat(1,N,1) if bert_src is not None else None,
                                           tgt_gpt = lm_tgt)
        (token_scores, next_copy_prob) = self.generator.predict(transformer_out, tgt_sequence_lengths)
        return (token_scores, next_copy_prob), src_attn_weights
