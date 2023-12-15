import math
import progressbar
import torch
from kinmt_decode_kin2en import kin2en_init_models, kin2en_translate
from misc_functions import read_lines


def translate_corpus(input_filename:str, output_filename:str, expected_length:int, parsed=True):
    from arguments import py_trainer_args
    rank = 0
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(rank)

    args = py_trainer_args(list_args=[], silent=True)

    args.load_saved_model = True
    args.kinmt_use_bert = True
    args.kinmt_use_gpt = False
    args.use_cross_positional_attn_bias = True
    args.kinmt_use_copy_loss = False
    args.kinmt_model_name = "kin2en_base_backtrans_bert_xpos_2023-05-02"

    kin2en_model_setup = kin2en_init_models(args, use_morpho_lib=(not parsed))

    print('Model: ready!')

    copy_threshold = 0.5
    intersect_threshold = 0.5
    beam_size = 4
    max_eval_sequences = 20
    counter = 0
    out_lines = []
    try:
        out_lines = read_lines(output_filename+".prev")
    except:
        pass
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        with open(input_filename, 'r', encoding='utf-8') as in_file:
            with progressbar.ProgressBar(max_value=(expected_length), redirect_stdout=True) as bar:
                for kinya_sentence in in_file:
                    if (counter % 100 == 0):
                        bar.update(counter)
                    if counter < len(out_lines):
                        out_file.write(f'{out_lines[counter]}\n')
                        out_file.flush()
                    else:
                        input_length = len(kinya_sentence.split('\t'))
                        max_text_length = math.ceil(input_length * 3)
                        (complete_translations,
                         pending_translations) = kin2en_translate(kin2en_model_setup,
                                                                  kinya_sentence,
                                                                  beam_size, max_eval_sequences, max_text_length,
                                                                  copy_threshold=copy_threshold,
                                                                  intersect_threshold=intersect_threshold)
                        kinmt_translation = kinya_sentence
                        if len(complete_translations) > 0:
                            kinmt_translation = complete_translations[0][1]
                        elif len(pending_translations) > 0:
                            kinmt_translation = pending_translations[0][1]

                        out_file.write(f'{kinmt_translation}\n')
                        out_file.flush()

                    counter += 1

if __name__ == '__main__':
    import sys
    key = sys.argv[1]
    translate_corpus(f'parsed_morpho_corpus_sentences_clean_2023-05-02_{key}_rw.txt',
                     f'morpho_corpus_sentences_clean_2023-05-02_{key}_en.txt',
                     500_005,
                     parsed = True)

    # servers = [('44110', 'root@173.49.207.77'),
    #            ('44110', 'root@173.49.207.77'),
    #            ('44110', 'root@173.49.207.77'),
    #            ('40396', 'root@172.74.91.206'),
    #            ('40396', 'root@172.74.91.206'),
    #            ('40396', 'root@172.74.91.206'),
    #            ('40373', 'root@172.74.91.206'),
    #            ('40373', 'root@172.74.91.206'),
    #            ('40373', 'root@172.74.91.206'),
    #            ('50186', 'root@75.174.234.163'),
    #            ('50186', 'root@75.174.234.163'),
    #            ('50186', 'root@75.174.234.163'),
    #            ('40153', 'root@136.228.125.173'),
    #            ('40153', 'root@136.228.125.173'),
    #            ('40153', 'root@136.228.125.173'),
    #            ('40132', 'root@136.228.125.173'),
    #            ('40132', 'root@136.228.125.173'),
    #            ('40132', 'root@136.228.125.173'),
    #            ('40119', 'root@136.228.125.173'),
    #            ('40119', 'root@136.228.125.173'),
    #            ('40119', 'root@136.228.125.173'),
    #            ('41074', 'root@24.17.114.242'),
    #            ('41074', 'root@24.17.114.242'),
    #            ('41074', 'root@24.17.114.242'),
    #            ('48138', 'root@50.5.250.227'),
    #            ('48138', 'root@50.5.250.227'),
    #            ('48138', 'root@50.5.250.227'),
    #            ('20685', 'root@206.75.146.212'),
    #            ('20685', 'root@206.75.146.212'),
    #            ('20685', 'root@206.75.146.212')]
    # end = -3_000_000
    # size = 500_000
    # start = end - size
    # idx = 0
    # while start >= -15_755_461:
    #     key = f'{-start//1_000}K-{-end//1_000}K'
    #     file = f'kinmt/parallel_data_2022/txt/parsed_morpho_corpus_sentences_clean_2023-05-02_{key}_rw.txt'
    #     scp = f'scp -i /home/user/myid.txt -P {servers[idx][0]} {file}.gz {servers[idx][1]}:'
    #     # print(f'tail -n {-start} parsed_morpho_corpus_sentences_clean_2023-05-02.txt | head -n {size} > {file}\ngzip {file}\n{scp}')
    #     port = servers[idx][0]
    #     machine = servers[idx][1]
    #     print(f'ssh -p {port} {machine}')
    #     print(f'gunzip parsed_morpho_corpus_sentences_clean_2023-05-02_{key}_rw.txt.gz')
    #     print(f'cd ')
    #     print(f'nohup python2 backtrans_kin2en.py {key} &>> /home/user/logs/{key}.log &')
    #     print(f'tail -f /home/user/logs/{key}.log\n\n\n')
    #     end -= size
    #     start = end - size
    #     idx += 1
