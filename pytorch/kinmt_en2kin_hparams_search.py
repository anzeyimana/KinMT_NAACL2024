import math
import progressbar
from torchmetrics import CHRFScore

from arguments import py_trainer_args
from kinmt_decode_en2kin import en2kin_init_models, en2kin_translate, read_english_lexicon
from misc_functions import read_lines, time_now


def en2kin_eval(args):

    en2kin_model_setup = en2kin_init_models(args)
    english_lexicon = read_english_lexicon(lexicon_file='english_lexicon.tsv')
    chrf = CHRFScore()

    for prob_cutoff in [0.7]:
        for affix_prob_cutoff in [0.9]:
            for affix_min_prob in [0.3]:
                for lprob_score_delta in [10.0]:
                    benchmarks = ['flores200_dev', 'mafand_mt_dev', 'tico19_dev']
                    overall_count = 0.0
                    overall_chrf2 = 0.0
                    for bench in benchmarks:
                        args.kinmt_eval_benchmark_name = bench
                        print(time_now(), 'Evaluating Model:' , args.kinmt_model_name, ' @ Benchmark:' , args.kinmt_eval_benchmark_name)
                        rw_lines = read_lines(f'kinmt/parallel_data_2022/txt/{args.kinmt_eval_benchmark_name}_rw.txt')
                        en_lines = read_lines(f'kinmt/parallel_data_2022/txt/{args.kinmt_eval_benchmark_name}_en.txt')
                        pairs = [(rw, en) for rw, en in zip(rw_lines, en_lines) if ((len(rw.split()) < 300) and (len(en.split()) < 300))]
                        kinmt_chrf = 0.0
                        total = 0.0
                        with progressbar.ProgressBar(max_value=(len(pairs)), redirect_stdout=True) as bar:
                            for idx, (kinya_sentence, engl_sentence) in enumerate(pairs):
                                bar.update(idx)
                                try:
                                    beam_size = 4
                                    morpho_synth_table_size = 6
                                    max_eval_sequences = 4
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
                                                                              prob_cutoff=prob_cutoff,
                                                                              affix_prob_cutoff=affix_prob_cutoff,
                                                                              affix_min_prob=affix_min_prob,
                                                                              lprob_score_delta=lprob_score_delta)
                                    kinmt_translation = engl_sentence
                                    if len(complete_translations) > 0:
                                        kinmt_translation = complete_translations[0][1]
                                    elif len(pending_translations) > 0:
                                        kinmt_translation = pending_translations[0][1]
                                    kinmt_score = chrf([kinmt_translation], [[kinya_sentence]]).item()
                                    kinmt_chrf += kinmt_score
                                    overall_chrf2 += kinmt_score
                                    overall_count += 1.0
                                    total += 1.0
                                except:
                                    print('Can\'t process due to potential large input' )

                        print(time_now(), 'Final CHRF2' ,
                              'Model:' , args.kinmt_model_name,
                              '@ Benchmark:' , args.kinmt_eval_benchmark_name,
                              'HP-Setup: prob_cutoff={}, affix_prob_cutoff: {}, affix_min_prob: {}, lprob_score_delta: {}'.format(prob_cutoff, affix_prob_cutoff, affix_min_prob, lprob_score_delta),
                              'KINMT: {:.2f}'.format(100.0 * kinmt_chrf / total))
                    print(time_now(), 'Finally CHRF2' ,
                          'Model:' , args.kinmt_model_name,
                          '@ Benchmarks:' , benchmarks,
                          'HP-Setup: prob_cutoff={}, affix_prob_cutoff: {}, affix_min_prob: {}, lprob_score_delta: {}'.format(prob_cutoff, affix_prob_cutoff, affix_min_prob, lprob_score_delta),
                          'KINMT: {:.3f}'.format(100.0 * overall_chrf2 / overall_count))

if __name__ == '__main__':
    import os
    args = py_trainer_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    if args.gpus == 0:
        args.world_size = 1
    en2kin_eval(args)
