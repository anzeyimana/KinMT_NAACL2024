from misc_functions import read_lines
from fairseq.models.roberta import RobertaModel
import progressbar

from datetime import datetime

english_BOS_idx = 0
english_EOS_idx = 2

def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def process_english_roberta_text_file(roberta, in_filename, out_filename):
    print(time_now(), 'Reading', in_filename, '...')
    lines = read_lines(in_filename)
    print(time_now(), 'Processing', in_filename, '...')
    out_file = open(out_filename, 'w')
    with progressbar.ProgressBar(max_value=(len(lines)), redirect_stdout=True) as bar:
        bar.update(0)
        i = 0
        for line in lines:
            tokens_line = ' '.join(line.replace('<unk>', ' ').replace('<s>', ' ').replace('</s>', ' ').replace('\t', ' ').split())
            token_ids = roberta.encode(tokens_line).cpu().numpy().tolist()
            if token_ids[0] != english_BOS_idx:
                token_ids = [english_BOS_idx] + token_ids
            if token_ids[-1] != english_EOS_idx:
                token_ids = token_ids + [english_EOS_idx]
            out_file.write('\t'.join([str(id) for id in token_ids]) + "\n")
            # line_ids = []
            # line_ids.append(('', [english_BOS_idx]))
            # for token in tokens:
            #     token_ids = roberta.encode(token).cpu().numpy().tolist()
            #     if token_ids[0] == english_BOS_idx:
            #         token_ids = token_ids[1:]
            #     if token_ids[-1] == english_EOS_idx:
            #         token_ids = token_ids[:-1]
            #     line_ids.append((token,token_ids))
            # line_ids.append(('', [english_EOS_idx]))
            # str_val = '\t'.join([(','.join([str(i) for i in ids]) + '|' + tk) for (tk,ids) in line_ids])
            # out_file.write(str_val + "\n")
            i += 1
            if ((i%10000) == 0):
                bar.update(i)
    out_file.close()

if __name__ == '__main__':
    roberta = RobertaModel.from_pretrained('roberta.base', checkpoint_file='model.pt')
    # items = ['jw', 'gazette', 'mafand_mt_dev','mafand_mt_test', 'tico19_dev', 'tico19_test', 'flores200_dev', 'flores200_test', 'habumuremyi_combined','kinyarwandanet', 'stage1', 'stage2', 'stage3', 'stage3_residuals', 'names_and_numbers', 'foreign_terms', 'numeric_examples']
    # items = ['morpho_corpus_sentences_clean_2023-05-02', 'june22_tmp_en2kin_backtrans']
    items = ['morpho_clean_corpus_2023-08-10']
    for it in items:
        process_english_roberta_text_file(roberta,
                                  f'kinmt/parallel_data_2022/txt/{it}_en.txt',
                                  f'kinmt/parallel_data_2022/txt/parsed_{it}_en.txt')
