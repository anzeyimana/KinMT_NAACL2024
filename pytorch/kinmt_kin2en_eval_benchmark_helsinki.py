import progressbar
from arguments import py_trainer_args
from kinmt_eval_bleurt import BLEURTScore
from kinmt_eval_helsinki_opus_mt import init_helsinki_opus_mt, helsinki_opus_mt_translate
from misc_functions import read_lines, time_now
from torchmetrics import CHRFScore


def kin2en_eval(args):
    helsinki_opus_mt_setup = init_helsinki_opus_mt(src_lang='rw', tgt_lang='en')
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
        helsinki_bleurt = 0.0
        helsinki_chrf = 0.0
        with progressbar.ProgressBar(max_value=(len(pairs)), redirect_stdout=True) as bar:
            for idx, (kinya_sentence, engl_sentence) in enumerate(pairs):
                bar.update(idx)
                try:
                    helsinki_translation = helsinki_opus_mt_translate(helsinki_opus_mt_setup, kinya_sentence)
                    helsinki_bleurt += bleurt([helsinki_translation], [engl_sentence])[0]
                    helsinki_chrf += chrf([helsinki_translation], [[engl_sentence]]).item()
                    total += 1
                except:
                    print('Can\'t process due to potential large input' )

        print(time_now(), 'Final BLEURT-20 & CHRF' ,
              'Model:' , args.kinmt_model_name,
              '@ Benchmark:' , args.kinmt_eval_benchmark_name,
              'HELSINKI: {:.2f}\t{:.2f}'.format(100.0 * helsinki_bleurt / total, 100.0 * helsinki_chrf / total))

if __name__ == '__main__':
    import os
    args = py_trainer_args(silent=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    if args.gpus == 0:
        args.world_size = 1
    kin2en_eval(args)
