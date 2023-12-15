import math

import progressbar
from arguments import py_trainer_args
from kinmt_decode_seq2seq import seq2seq_init_models, seq2seq_translate
from kinmt_eval_bleurt import BLEURTScore
from misc_functions import read_lines, time_now
from torchmetrics import CHRFScore

def seq2seq_eval(args):
    model_setup = seq2seq_init_models(args)

    chrf = CHRFScore()
    bleurt = BLEURTScore(model_name='lucadiliello/BLEURT-20')

    benchmarks = ['flores200_dev', 'flores200_test', 'mafand_mt_dev', 'mafand_mt_test', 'tico19_dev', 'tico19_test']
    for bench in benchmarks:
        args.kinmt_eval_benchmark_name = bench
        print(time_now(), 'Evaluating Model:' , args.kinmt_model_name, args.kinmt_seq2seq_config, ' @ Benchmark:' , args.kinmt_eval_benchmark_name)
        rw_lines = read_lines(f'kinmt/parallel_data_2022/txt/{args.kinmt_eval_benchmark_name}_rw.txt')
        en_lines = read_lines(f'kinmt/parallel_data_2022/txt/{args.kinmt_eval_benchmark_name}_en.txt')
        pairs = [(rw, en) for rw, en in zip(rw_lines, en_lines) if ((len(rw.split()) < 300) and (len(en.split()) < 300))]

        kinmt_bleurt = 0.0
        kinmt_chrf = 0.0

        # helsinki_bleurt = 0.0
        # helsinki_chrf = 0.0

        kinmt_total = 0.0
        # helsinki_total = 0.0
        with progressbar.ProgressBar(max_value=(len(pairs)), redirect_stdout=True) as bar:
            for idx, (kin_sentence, en_sentence) in enumerate(pairs):
                if args.kinmt_seq2seq_config == 'kin2en':
                    src_sentence, tgt_sentence = kin_sentence, en_sentence
                else:
                    src_sentence, tgt_sentence = en_sentence, kin_sentence
                if (idx % 100) == 0:
                    bar.update(idx)
                # if True:
                try:
                    beam_size = 4
                    max_eval_sequences = 8
                    input_length = len(src_sentence.split(' '))
                    max_text_length = math.ceil(input_length * 4)

                    (complete_translations,
                     pending_translations) = seq2seq_translate(model_setup, src_sentence,
                                                               beam_size, max_eval_sequences, max_text_length)

                    kinmt_translation = src_sentence
                    if len(complete_translations) > 0:
                        kinmt_translation = complete_translations[0][1]
                    elif len(pending_translations) > 0:
                        kinmt_translation = pending_translations[0][1]

                    if args.kinmt_seq2seq_config == 'kin2en':
                        kinmt_bleurt += bleurt([kinmt_translation], [tgt_sentence])[0]
                    else:
                        kinmt_bleurt += 1.0
                    kinmt_chrf += chrf([kinmt_translation], [[tgt_sentence]]).item()
                    kinmt_total += 1

                except Exception as error:
                    print("An error occurred:", type(error).__name__, "–", error)  # An error occurred: NameError – name 'x' is not defined

        print(time_now(), 'Final BLEURT-20 & CHRF' ,
              'Model:' , args.kinmt_model_name,
              '@ Benchmark:' , args.kinmt_eval_benchmark_name,
              'KINMT: {:.2f}\t{:.2f}'.format(100.0 * kinmt_bleurt / kinmt_total, 100.0 * kinmt_chrf / kinmt_total))

if __name__ == '__main__':
    import os
    args = py_trainer_args(silent=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    if args.gpus == 0:
        args.world_size = 1
    seq2seq_eval(args)
