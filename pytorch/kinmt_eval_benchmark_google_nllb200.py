import progressbar
from torchmetrics import CHRFScore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from arguments import py_trainer_args
from kinmt_eval_bleurt import BLEURTScore
from kinmt_eval_google_translate import init_google_translate_client, gogle_translate
from misc_functions import read_lines, time_now


def init_nllb200(device=0):
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    mt_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    kin2en_translator = pipeline('translation',
                                 model=mt_model,
                                 tokenizer=tokenizer,
                                 src_lang='kin_Latn', tgt_lang='eng_Latn', max_length = 400, device=device)
    en2kin_translator = pipeline('translation',
                                 model=mt_model,
                                 tokenizer=tokenizer,
                                 src_lang='eng_Latn', tgt_lang='kin_Latn', max_length = 400, device=device)
    return (kin2en_translator, en2kin_translator)

def nllb200_kin2en_translate(nllb200_model_setup, text:str) -> str:
    (kin2en_translator, en2kin_translator) = nllb200_model_setup
    translation = kin2en_translator([text])[0]['translation_text']
    return translation

def nllb200_en2kin_translate(nllb200_model_setup, text:str) -> str:
    (kin2en_translator, en2kin_translator) = nllb200_model_setup
    translation = en2kin_translator([text])[0]['translation_text']
    return translation

def kinmt_eval(args):
    nllb200_model_setup = init_nllb200()
    google_client = init_google_translate_client()
    chrf = CHRFScore()
    bleurt = BLEURTScore(model_name='lucadiliello/BLEURT-20')

    benchmarks = ['flores200_dev', 'flores200_test', 'mafand_mt_dev', 'mafand_mt_test', 'tico19_dev', 'tico19_test']
    for bench in benchmarks:
        args.kinmt_eval_benchmark_name = bench
        print(time_now(), 'Evaluating Model:' , args.kinmt_model_name, ' @ Benchmark:' , args.kinmt_eval_benchmark_name)
        rw_lines = read_lines(f'kinmt/parallel_data_2022/txt/{args.kinmt_eval_benchmark_name}_rw.txt')
        en_lines = read_lines(f'kinmt/parallel_data_2022/txt/{args.kinmt_eval_benchmark_name}_en.txt')
        pairs = [(rw, en) for rw, en in zip(rw_lines, en_lines) if ((len(rw.split()) < 300) and (len(en.split()) < 300))]
        total = 0.0
        google_kin2en_bleurt = 0.0
        google_kin2en_chrf = 0.0
        google_en2kin_chrf = 0.0

        nllb_kin2en_bleurt = 0.0
        nllb_kin2en_chrf = 0.0
        nllb_en2kin_chrf = 0.0

        with progressbar.ProgressBar(max_value=(len(pairs)), redirect_stdout=True) as bar:
            for idx, (kinya_sentence, engl_sentence) in enumerate(pairs):
                bar.update(idx)
                try:
                    google_english = gogle_translate(google_client, [kinya_sentence], src_lang='rw', tgt_lang='en')[0]
                    google_kinya = gogle_translate(google_client, [engl_sentence], src_lang='en', tgt_lang='rw')[0]
                    nllb_english = nllb200_kin2en_translate(nllb200_model_setup, kinya_sentence)
                    nllb_kinya = nllb200_en2kin_translate(nllb200_model_setup, engl_sentence)

                    google_kin2en_bleurt_ = bleurt([google_english], [engl_sentence])[0]
                    google_kin2en_chrf_ = chrf([google_english], [[engl_sentence]]).item()
                    google_en2kin_chrf_ = chrf([google_kinya], [[kinya_sentence]]).item()

                    nllb_kin2en_bleurt_ = bleurt([nllb_english], [engl_sentence])[0]
                    nllb_kin2en_chrf_ = chrf([nllb_english], [[engl_sentence]]).item()
                    nllb_en2kin_chrf_ = chrf([nllb_kinya], [[kinya_sentence]]).item()

                    total += 1

                    google_kin2en_bleurt += google_kin2en_bleurt_
                    google_kin2en_chrf += google_kin2en_chrf_
                    google_en2kin_chrf += google_en2kin_chrf_

                    nllb_kin2en_bleurt += nllb_kin2en_bleurt_
                    nllb_kin2en_chrf += nllb_kin2en_chrf_
                    nllb_en2kin_chrf += nllb_en2kin_chrf_

                except:
                    print('Can\'t process probably due to potential large input or other error' )

        print(time_now(), 'Final BLEURT-20 & CHRF' ,
              '@ Benchmark:' , args.kinmt_eval_benchmark_name,
              'GOOGLE KIN2EN: {:.2f}\t{:.2f}'.format(100.0 * google_kin2en_bleurt / total, 100.0 * google_kin2en_chrf / total))
        print(time_now(), 'Final CHRF' ,
              '@ Benchmark:' , args.kinmt_eval_benchmark_name,
              'GOOGLE EN2KIN: {:.2f}'.format(100.0 * google_en2kin_chrf / total))

        print(time_now(), 'Final BLEURT-20 & CHRF' ,
              '@ Benchmark:' , args.kinmt_eval_benchmark_name,
              'NLLB KIN2EN: {:.2f}\t{:.2f}'.format(100.0 * nllb_kin2en_bleurt / total, 100.0 * nllb_kin2en_chrf / total))
        print(time_now(), 'Final CHRF' ,
              '@ Benchmark:' , args.kinmt_eval_benchmark_name,
              'NLLB EN2KIN: {:.2f}'.format(100.0 * nllb_en2kin_chrf / total))

if __name__ == '__main__':
    import os
    args = py_trainer_args(silent=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    if args.gpus == 0:
        args.world_size = 1
    kinmt_eval(args)
