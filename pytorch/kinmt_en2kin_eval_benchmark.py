import math

import progressbar
from torchmetrics import CHRFScore

import morpho_model

from arguments import py_trainer_args
from kinmt_decode_en2kin import en2kin_init_models, en2kin_translate, read_english_lexicon
from misc_functions import read_lines, time_now
# from kinmt_eval_helsinki_opus_mt import init_helsinki_opus_mt, helsinki_opus_mt_translate


def en2kin_eval(args):
    morpho_model.SOCKET_SYNTHESIS = False

    en2kin_model_setup = en2kin_init_models(args, init_morphoy=True, socket_synthesis=False)

    english_lexicon = read_english_lexicon(lexicon_file='english_lexicon.tsv')
    # helsinki_opus_mt_setup = init_helsinki_opus_mt(src_lang='en', tgt_lang='rw')
    chrf = CHRFScore()

    benchmarks = ['flores200_dev', 'flores200_test', 'mafand_mt_dev', 'mafand_mt_test', 'tico19_dev', 'tico19_test']
    # benchmarks = ['flores200_dev', 'mafand_mt_dev', 'tico19_dev']
    for bench in benchmarks:
        args.kinmt_eval_benchmark_name = bench
        print(time_now(), 'Evaluating Model:' , args.kinmt_model_name, ' @ Benchmark:' , args.kinmt_eval_benchmark_name)
        rw_lines = read_lines(f'kinmt/parallel_data_2022/txt/{args.kinmt_eval_benchmark_name}_rw.txt')
        en_lines = read_lines(f'kinmt/parallel_data_2022/txt/{args.kinmt_eval_benchmark_name}_en.txt')
        pairs = [(rw, en) for rw, en in zip(rw_lines, en_lines) if ((len(rw.split()) < 300) and (len(en.split()) < 300))]
        # helsinki_chrf = 0.0
        kinmt_chrf = 0.0
        total = 0.0
        with progressbar.ProgressBar(max_value=(len(pairs)), redirect_stdout=True) as bar:
            for idx, (kinya_sentence, engl_sentence) in enumerate(pairs):
                if (idx % 100) == 0:
                    bar.update(idx)
                if True:
                # try:
                    beam_size = 4
                    morpho_synth_table_size = 8
                    max_eval_sequences = 6
                    input_length = len(engl_sentence.split())
                    max_text_length = math.ceil(input_length * 3)

                    (complete_translations,
                     pending_translations) = en2kin_translate(en2kin_model_setup,
                                                              engl_sentence,
                                                              max_text_length,
                                                              max_eval_sequences,
                                                              max_morpho_inference_table_length=morpho_synth_table_size,
                                                              max_batch_size=beam_size, ENGL_SEGMENT_PER_TOKEN=args.kinmt_engl_parse_per_token,
                                                              english_lexicon = english_lexicon,
                                                              prob_cutoff=args.kinmt_en2kin_prob_cutoff,
                                                              affix_prob_cutoff=args.kinmt_en2kin_affix_prob_cutoff,
                                                              affix_min_prob=args.kinmt_en2kin_affix_min_prob,
                                                              lprob_score_delta=args.kinmt_en2kin_lprob_score_delta,
                                                              use_bert=args.kinmt_use_bert)
                    # helsinki_translation = helsinki_opus_mt_translate(helsinki_opus_mt_setup, engl_sentence)
                    kinmt_translation = engl_sentence
                    if len(complete_translations) > 0:
                        kinmt_translation = complete_translations[0][1]
                    elif len(pending_translations) > 0:
                        kinmt_translation = pending_translations[0][1]
                    # helsinki_score = chrf([helsinki_translation], [[kinya_sentence]]).item()
                    kinmt_score = chrf([kinmt_translation], [[kinya_sentence]]).item()
                    # print(kinmt_score,f'\t\'{kinmt_translation}\'\t\'{kinya_sentence}\'',flush=True)
                    kinmt_chrf += kinmt_score
                    # helsinki_chrf += helsinki_score
                    total += 1
                # except Exception as error:
                #     print("An error occurred:", type(error).__name__, "–", error)  # An error occurred: NameError – name 'x' is not defined

        print(time_now(), 'Final CHRF2' ,
              'Model:' , args.kinmt_model_name,
              '@ Benchmark:' , args.kinmt_eval_benchmark_name,
              'KINMT: {:.2f}'.format(100.0 * kinmt_chrf / total))
        # print(time_now(), 'Final CHRF' ,
        #       '@ Benchmark:' , args.kinmt_eval_benchmark_name,
        #       'HELSINKI: {:.2f}'.format(100.0 * helsinki_chrf / total))

if __name__ == '__main__':
    import os
    args = py_trainer_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    if args.gpus == 0:
        args.world_size = 1
    en2kin_eval(args)
