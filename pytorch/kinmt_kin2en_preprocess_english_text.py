from datetime import datetime

import progressbar

from fairseq.models.transformer_lm import TransformerLanguageModel

eng_lm_BOS_idx = 0
eng_lm_EOS_idx = 2

def read_lines(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    lines = [line.rstrip('\n') for line in f]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    f.close()
    return lines
def write_lines(lines, file_name):
    f = open(file_name, 'w', encoding='utf-8')
    for l in lines:
        f.write(l+'\n')
    f.close()

def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def process_english_lm_text_file(english_lm, in_filename, out_filename):
    print(time_now(), 'Reading', in_filename, '...')
    lines = read_lines(in_filename)
    print(time_now(), 'Processing', in_filename, '...')
    out_file = open(out_filename, 'w')
    with progressbar.ProgressBar(max_value=(len(lines)), redirect_stdout=True) as bar:
        bar.update(0)
        i = 0
        for line in lines:
            tokens_line = ' '.join(line.replace('<unk>', ' ').replace('<s>', ' ').replace('</s>', ' ').replace('\t', ' ').split())
            token_ids = english_lm.encode(tokens_line).cpu().numpy().tolist()
            if token_ids[0] != eng_lm_BOS_idx:
                token_ids = [eng_lm_BOS_idx] + token_ids
            if token_ids[-1] != eng_lm_EOS_idx:
                token_ids = token_ids + [eng_lm_EOS_idx]
            out_file.write('\t'.join([str(id) for id in token_ids]) + "\n")
            # tokens = ' '.join(line.replace('\t', ' ').split()).split()
            # line_ids = []
            # line_ids.append(('', [eng_lm_BOS_idx]))
            # for token in tokens:
            #     token_ids = english_lm.encode(token).cpu().numpy().tolist()
            #     if token_ids[0] == eng_lm_BOS_idx:
            #         token_ids = token_ids[1:]
            #     if token_ids[-1] == eng_lm_EOS_idx:
            #         token_ids = token_ids[:-1]
            #     line_ids.append((token, token_ids))
            # line_ids.append(('', [eng_lm_EOS_idx]))
            # str_val = '\t'.join([(','.join([str(i) for i in ids]) + '|' + tk) for (tk,ids) in line_ids])
            # out_file.write(str_val + "\n")
            i += 1
            if ((i%1000) == 0):
                bar.update(i)
    out_file.close()

if __name__ == '__main__':
    english_lm = TransformerLanguageModel.from_pretrained('wmt19.en', 'model.pt', tokenizer='moses', bpe='fastbpe').cpu()
    # items = ['jw', 'gazette', 'mafand_mt_dev','mafand_mt_test', 'tico19_dev', 'tico19_test', 'flores200_dev', 'flores200_test', 'habumuremyi_combined','kinyarwandanet', 'stage1', 'stage2', 'stage3', 'stage3_residuals', 'names_and_numbers', 'foreign_terms', 'numeric_examples']
    # items = ['morpho_corpus_sentences_clean_2023-05-02', 'june21_tmp_en2kin_backtrans']
    items = ['june27_tmp_en2kin_backtrans']
    for it in items:
        process_english_lm_text_file(english_lm,
                                  f'kinmt/parallel_data_2022/txt/{it}_en.txt',
                                  f'kinmt/parallel_data_2022/txt/parsed_{it}_lm_en.txt')
