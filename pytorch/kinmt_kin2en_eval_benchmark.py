import math

import progressbar
import torch
from torchmetrics import CHRFScore
from transformers import MarianMTModel, MarianTokenizer

from arguments import py_trainer_args
from kinmt_decode_kin2en import kin2en_init_models, kin2en_translate
from kinmt_eval_bleurt import BLEURTScore
from misc_functions import read_lines, time_now

def init_ft_helsinki():
    device = torch.device('cuda:0')
    model_name = 'opus-mt-rw-en-finetuned-rw-to-en/checkpoint-1380000'
    tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()
    return (model, tokenizer)

def ft_helnksi_translate(model_setup, src_text: str):
    device = torch.device('cuda:0')
    (model, tokenizer) = model_setup
    with torch.no_grad():
        translated = model.generate(**tokenizer([src_text], return_tensors="pt", padding=True).to(device))
        return tokenizer.decode(translated[0], skip_special_tokens=True)

def kin2en_eval(args):
    # args.kinmt_model_name = args.kinmt_model_name+".pt_best_valid_loss"
    # ft_helsinki_model_setup = init_ft_helsinki()

    kin2en_model_setup = kin2en_init_models(args, use_morpho_lib=False)
    # kin2en_model_setup = kin2en_init_models(args, use_morpho_lib=True)

    chrf = CHRFScore()
    bleurt = BLEURTScore(model_name='lucadiliello/BLEURT-20')

    benchmarks = ['flores200_dev', 'flores200_test', 'mafand_mt_dev', 'mafand_mt_test', 'tico19_dev', 'tico19_test']
    for bench in benchmarks:
        args.kinmt_eval_benchmark_name = bench
        print(time_now(), 'Evaluating Model:' , args.kinmt_model_name, ' @ Benchmark:' , args.kinmt_eval_benchmark_name)
        rw_lines = read_lines(f'kinmt/parallel_data_2022/txt/parsed_{args.kinmt_eval_benchmark_name}_rw.txt')
        # rw_lines = read_lines(f'kinmt/parallel_data_2022/txt/{args.kinmt_eval_benchmark_name}_rw.txt')
        en_lines = read_lines(f'kinmt/parallel_data_2022/txt/{args.kinmt_eval_benchmark_name}_en.txt')
        pairs = [(rw, en) for rw, en in zip(rw_lines, en_lines) if ((len(rw.split()) < 300) and (len(en.split()) < 300))]

        kinmt_bleurt = 0.0
        kinmt_chrf = 0.0

        # helsinki_bleurt = 0.0
        # helsinki_chrf = 0.0

        kinmt_total = 0.0
        # helsinki_total = 0.0
        with progressbar.ProgressBar(max_value=(len(pairs)), redirect_stdout=True) as bar:
            for idx, (kinya_sentence, engl_sentence) in enumerate(pairs):
                if (idx % 100) == 0:
                    bar.update(idx)
                # if True:
                try:
                    copy_threshold = 0.5
                    intersect_threshold = 0.5
                    beam_size = 8
                    max_eval_sequences = 10
                    input_length = len(kinya_sentence.split(';'))
                    # input_length = len(kinya_sentence.split())
                    max_text_length = math.ceil(input_length * 3)

                    (complete_translations,
                     pending_translations) = kin2en_translate(kin2en_model_setup,
                                                              kinya_sentence,
                                                              beam_size, max_eval_sequences, max_text_length,
                                                              copy_threshold=copy_threshold,
                                                              intersect_threshold=intersect_threshold,
                                                              attempt_auto_correction=False)

                    # helsinki_translation = ft_helnksi_translate(ft_helsinki_model_setup, kinya_sentence)

                    kinmt_translation = kinya_sentence
                    if len(complete_translations) > 0:
                        kinmt_translation = complete_translations[0][1]
                    elif len(pending_translations) > 0:
                        kinmt_translation = pending_translations[0][1]

                    kinmt_bleurt += bleurt([kinmt_translation], [engl_sentence])[0]
                    kinmt_chrf += chrf([kinmt_translation], [[engl_sentence]]).item()
                    kinmt_total += 1

                    # helsinki_bleurt += bleurt([helsinki_translation], [engl_sentence])[0]
                    # helsinki_chrf += chrf([helsinki_translation], [[engl_sentence]]).item()
                    # helsinki_total += 1
                except Exception as error:
                    print("An error occurred:", type(error).__name__, "–", error)  # An error occurred: NameError – name 'x' is not defined

        print(time_now(), 'Final BLEURT-20 & CHRF' ,
              'Model:' , args.kinmt_model_name,
              '@ Benchmark:' , args.kinmt_eval_benchmark_name,
              'KINMT: {:.2f}\t{:.2f}'.format(100.0 * kinmt_bleurt / kinmt_total, 100.0 * kinmt_chrf / kinmt_total))

        # print(time_now(), 'Final BLEURT-20 & CHRF' ,
        #       'Model:' , args.kinmt_model_name,
        #       '@ Benchmark:' , args.kinmt_eval_benchmark_name,
        #       'HELSINKI: {:.2f}\t{:.2f}'.format(100.0 * helsinki_bleurt / helsinki_total, 100.0 * helsinki_chrf / helsinki_total))

if __name__ == '__main__':
    import os
    args = py_trainer_args(silent=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    if args.gpus == 0:
        args.world_size = 1
    kin2en_eval(args)
