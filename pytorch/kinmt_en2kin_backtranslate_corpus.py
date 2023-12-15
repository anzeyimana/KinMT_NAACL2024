import math
import progressbar
import torch
from arguments import py_trainer_args
from kinmt_decode_en2kin import en2kin_init_models, read_english_lexicon, en2kin_translate

def trigger_server_report(en2kin_model_setup):
    (en2kin_model, engl_roberta_model, kin_lm_model, ffi, lib, device,
     stems_vocab, all_affixes, all_afsets, all_afsets_inverted_index,
     afset_affix_corr, afset_stem_corr, pos_afset_corr, pos_stem_corr,
     afset_affix_slot_corr) = en2kin_model_setup
    lib.trigger_server_report()

def translate_english_corpus(args, input_filename:str, output_filename:str, expected_length:int):
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    rank = ((args.corpus_id - 1) % 8)
    torch.cuda.set_device(rank)

    args.load_saved_model = True
    args.kinmt_use_bert = True
    args.kinmt_use_gpt = False
    args.use_cross_positional_attn_bias = True
    args.kinmt_use_copy_loss = False
    args.kinmt_model_name = "en2kin_base_final_back_trans_eval_bert_xpos_2023-06-08.pt_ep1_121K"

    en2kin_model_setup = en2kin_init_models(args, rank=rank, socket_synthesis=True, morpho_conf = f'config_morpho_{rank}.conf')
    english_lexicon = read_english_lexicon(lexicon_file='english_lexicon.tsv')

    beam_size = 4
    morpho_synth_table_size = 6
    max_eval_sequences = 4

    counter = 0
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        with open(input_filename, 'r', encoding='utf-8') as in_file:
            with progressbar.ProgressBar(max_value=(expected_length), redirect_stdout=True) as bar:
                for engl_sentence in in_file:
                    engl_sentence = engl_sentence.strip('\n').rstrip('\n').strip().rstrip()
                    if len(engl_sentence) > 0:
                        if (counter % 100 == 0):
                            bar.update(counter)
                            trigger_server_report(en2kin_model_setup)

                        input_length = len(engl_sentence.split())
                        max_text_length = math.ceil(input_length * 3)

                        (complete_translations,
                         pending_translations) = en2kin_translate(en2kin_model_setup,
                                                                  engl_sentence,
                                                                  max_text_length,
                                                                  max_eval_sequences,
                                                                  max_morpho_inference_table_length=morpho_synth_table_size,
                                                                  max_batch_size=beam_size,
                                                                  ENGL_SEGMENT_PER_TOKEN=args.kinmt_engl_parse_per_token,
                                                                  english_lexicon=english_lexicon,
                                                                  prob_cutoff=args.kinmt_en2kin_prob_cutoff,
                                                                  affix_prob_cutoff=args.kinmt_en2kin_affix_prob_cutoff,
                                                                  affix_min_prob=args.kinmt_en2kin_affix_min_prob,
                                                                  lprob_score_delta=args.kinmt_en2kin_lprob_score_delta)
                        kinmt_translation = engl_sentence
                        if len(complete_translations) > 0:
                            kinmt_translation = complete_translations[0][1]
                        elif len(pending_translations) > 0:
                            kinmt_translation = pending_translations[0][1]

                        out_file.write(f'{kinmt_translation}\n')
                        out_file.flush()

                        counter += 1

if __name__ == '__main__':
    args = py_trainer_args()
    input_filename = f'kinmt/parallel_data_2022/txt/english_corpus_2023-06-10_P{args.corpus_id}_en.txt'
    output_filename = f'kinmt/parallel_data_2022/txt/english_corpus_2023-06-10_P{args.corpus_id}_rw.txt'
    expected_length = 250_000
    translate_english_corpus(args, input_filename, output_filename, expected_length)