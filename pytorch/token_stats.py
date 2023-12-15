from typing import Dict, List
import math
from misc_functions import read_lines

# id,correctness,totalCount,totalDocuments,allUpCount,allLowCount,firstUpCount,isNumeric
class TokenStats:
    def __init__(self, line:str, id: int, tot:int):
        toks = line.split(',')
        if (len(toks) == 8):
            self.id = toks[0]
        elif (len(toks) == 9):
            self.id = ','
        self.correctness = int(toks[-7])
        self.totalCount = float(toks[-6])
        self.totalDocuments = float(toks[-5])
        self.allUpCount = float(toks[-4])
        self.allLowCount = float(toks[-3])
        self.firstUpCount = float(toks[-2])
        self.isNumeric = int(toks[-1]) != 0
        self.pct_rank = int(100.0 * float(id) /float(tot))

def get_all_token_stats(all_tokens_file = "AllTokens.csv"):
    global all_tokens_stats
    lines = read_lines(all_tokens_file)
    tokens = [TokenStats(line,id,len(lines)) for id,line in enumerate(lines)]
    return {t.id:t for t in tokens}

def sort_keywords(token_stats: Dict[str,TokenStats], orig_tokens: List[str]) -> List[str]:
    lower_tokens = []
    for token in orig_tokens:
        key = token.lower()
        if (len(token) > 1) and ((token[-1] == '\'') or (token[-1] == '’') or (token[-1] == '’') or (token[-1] == '`')):
            key = token[:-1]+'a'
        lower_tokens.append(key)

    token_doc_counts = {tk:sum([1 for v in lower_tokens if v == tk]) for tk in lower_tokens}
    token_tf_idf = {}
    for key in lower_tokens:
        # REF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
        f = token_doc_counts[key]
        max_f = max([token_doc_counts[k] for k in token_doc_counts])
        tf = 0.5 + (0.5 * (f/max_f))
        total_docs = 1_000_000
        if key in token_stats:
            k_docs = token_stats[key].totalDocuments
        else:
            k_docs = 0.0
        idf = math.log(total_docs / (1.0 + k_docs))
        token_tf_idf[key] = tf * idf

    return sorted(lower_tokens, key=lambda x: token_tf_idf[x], reverse=True)






