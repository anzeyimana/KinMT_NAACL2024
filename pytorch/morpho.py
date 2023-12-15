from typing import List

from cffi import FFI
import youtokentome as yttm

from language_data import all_pos_tags
from token_stats import get_all_token_stats

PAD_ID = 0
UNK_ID = 1
MSK_ID = 2
BOS_ID = 3
EOS_ID = 4

NUM_SPECIAL_TOKENS = 5

clib_all_token_stats = None
def build_morphoy_lib():
    global clib_all_token_stats
    if clib_all_token_stats is None:
        clib_all_token_stats = get_all_token_stats()
    ffibuilder = FFI()

    ffibuilder.cdef("""

        typedef struct _morphoy_parsed_word_t {
            // POS Info
            int lm_stem_id;
            int lm_morph_id;
            int pos_tag_id;
    
            // Morphology
            int stem_id;
            int * affix_ids;
            int * extra_stem_token_ids;
            int len_affix_ids;
            int len_extra_stem_token_ids;
    
            // Text
            int uses_bpe;
            char * surface_form;
            char * raw_surface_form;
            int surface_form_has_valid_orthography;
            int len_surface_form;
            int len_raw_surface_form;
            int is_apostrophed;
            struct _morphoy_parsed_word_t * next;
        } morphoy_parsed_word_t;
    
        typedef struct _morphoy_parsed_sentence_t {
            morphoy_parsed_word_t * first_word;
            morphoy_parsed_word_t * last_word;
            int words_len;
            struct _morphoy_parsed_sentence_t * next_sent;
        } morphoy_parsed_sentence_t;
    
        typedef struct _morphoy_model_params {
            int tot_num_lm_stems;
            int tot_num_lm_morphs;
            int tot_num_pos_tags;
            int tot_num_stems;
            int tot_num_affixes;
        } morphoy_model_params_t;

        morphoy_parsed_sentence_t * morphoy_parse_text(const char * text, int num_sent[1], int use_mgt_machinery_if_no_saved_exists);
    
        void morphoy_delete_sentences(morphoy_parsed_sentence_t * sentences);
    
        morphoy_parsed_sentence_t * morphoy_parse_sentence(void * sentence, int use_mgt_machinery_if_no_saved_exists);
    
        morphoy_model_params_t * morphoy_get_model_params(void);

        void start_morpho_lib(const char * config_file);
    
        void stop_morpho_lib(void);
    
        void start_morpho_light_lib(const char * config_file);
    
        void stop_morpho_light_lib(void);
        
        char * synth_morpho_token(int wt_idx, const char * stem, const char * fsa_key, const char * indices_csv);
        
        void free_token(char * token);
        
        void init_morpho_synth_via_socket(const char * config_file);

        char * synth_morpho_token_via_socket(const char * wt_idx, const char * stem, const char * fsa_key, const char * indices_csv);
        
        void trigger_server_report(void);

    """)

    ffibuilder.set_source("morphoy",
                          """
                               #include "lib.h"
                               #include "morphoy.h"
                          """,
                          extra_compile_args=['-fopenmp', '-D use_openmp', '-O3', '-march=native', '-ffast-math',
                                              '-Wall', '-Werror'],
                          extra_link_args=['-fopenmp'],
                          libraries=['morpho'])  # library name, for the linker

    ffibuilder.compile(verbose=True)


class ParsedToken:
    def __init__(self, w, ffi):
        # POS Info
        self.lm_stem_id = w.lm_stem_id
        self.lm_morph_id = w.lm_morph_id
        self.pos_tag_id = w.pos_tag_id
        self.valid_orthography = w.surface_form_has_valid_orthography

        # Morphology
        self.stem_id = w.stem_id
        self.affix_ids = [w.affix_ids[i] for i in range(w.len_affix_ids)]
        self.extra_stem_token_ids = [w.extra_stem_token_ids[i] for i in range(w.len_extra_stem_token_ids)]

        # Text
        self.uses_bpe = (w.uses_bpe == 1)
        self.is_apostrophed = w.is_apostrophed
        self.surface_form = ffi.string(w.surface_form).decode("utf-8") if (w.len_surface_form > 0) else ''
        self.raw_surface_form = ffi.string(w.raw_surface_form).decode("utf-8") if (w.len_raw_surface_form > 0) else ''
        if (self.is_apostrophed != 0) and (len(self.raw_surface_form) > 0) and ((self.raw_surface_form[-1] == 'a') or (self.raw_surface_form[-1] == 'A')):
            self.raw_surface_form = self.raw_surface_form[:-1]+"\'"

    def to_parsed_format(self) -> str:
        word_list = [self.lm_stem_id, self.lm_morph_id, self.pos_tag_id, self.stem_id] + [len(self.extra_stem_token_ids)] + self.extra_stem_token_ids + [len(self.affix_ids)] + self.affix_ids
        return ','.join([str(x) for x in word_list])

class ParsedMorphoToken:
    def __init__(self, parsed_token, real_parsed_token : ParsedToken = None):
        if real_parsed_token is not None:
            self.lm_stem_id = real_parsed_token.lm_stem_id
            self.lm_morph_id = real_parsed_token.lm_morph_id
            self.pos_tag_id = real_parsed_token.pos_tag_id
            self.stem_id = real_parsed_token.stem_id # My God: This bug here almost caused me panic! lm_stem_id should be different from stem_id
            self.extra_tokens_ids = real_parsed_token.extra_stem_token_ids
            self.affixes = real_parsed_token.affix_ids
            self.is_apostrophed = real_parsed_token.is_apostrophed
            self.uses_bpe = real_parsed_token.uses_bpe
            self.surface_form = real_parsed_token.surface_form
            self.raw_surface_form = real_parsed_token.raw_surface_form
        else:
            self.uses_bpe = False
            self.surface_form = '_'
            self.raw_surface_form = '_'
            if '|' in parsed_token:
                idx = parsed_token.index('|')
                tks = parsed_token[:idx].split(',')
                sfc = parsed_token[(idx + 1):]
                if len(sfc) > 0:
                    self.surface_form = sfc.lower()
                    self.raw_surface_form = sfc
                    self.uses_bpe = True
            else:
                tks = parsed_token.split(',')
            self.lm_stem_id = int(tks[0])
            self.lm_morph_id = int(tks[1])
            self.pos_tag_id = int(tks[2])
            self.stem_id = int(tks[3])
            num_ext = int(tks[4])
            self.extra_tokens_ids = [int(v) for v in tks[5:(5+num_ext)]]
            # This is to cap too long tokens for position encoding
            self.extra_tokens_ids = self.extra_tokens_ids[:64]
            num_afx = int(tks[(5+num_ext)])
            self.affixes = [int(v) for v in tks[(6+num_ext):(6+num_ext+num_afx)]]
            self.is_apostrophed = 0

    def to_parsed_format(self) -> str:
        word_list = [self.lm_stem_id, self.lm_morph_id, self.pos_tag_id, self.stem_id] + [len(self.extra_tokens_ids)] + self.extra_tokens_ids + [len(self.affixes)] + self.affixes
        return ','.join([str(x) for x in word_list])
    def get_surface_forms(self, bpe:yttm.BPE):
        if len(self.extra_tokens_ids) > 0:
            tkns = bpe.encode([self.raw_surface_form],
                              output_type=yttm.OutputType.SUBWORD, bos=False, eos=False, reverse=False,
                             dropout_prob=0)[0]
            return [(k if (k[0] == '▁') else ('@@' + k)) for k in tkns]
        else:
            return [self.raw_surface_form]


class ParsedMorphoSentence:
    def __init__(self, parsed_sentence_line, parsed_tokens: List[ParsedToken] = None, delimeter=';'):
        if parsed_tokens is not None:
            self.tokens = [ParsedMorphoToken('_', real_parsed_token=token) for token in parsed_tokens]
        else:
            self.tokens = [ParsedMorphoToken(v) for v in parsed_sentence_line.split(delimeter) if len(v)>0]
    def to_parsed_format(self) -> str:
        return ';'.join([tk.to_parsed_format() for tk in self.tokens])

def is_rare_kinya_token(pt:ParsedToken) -> bool:
    global clib_all_token_stats
    if (pt.valid_orthography == 0):
        return False
    if (not (((pt.pos_tag_id - 5) < len(all_pos_tags)) and (pt.pos_tag_id >= 5))) and (pt.valid_orthography > 0):
        return True
    if not (pt.surface_form in clib_all_token_stats) and ((all_pos_tags[pt.pos_tag_id - 5]["name"]) == "N"):
        return True
    rank = (clib_all_token_stats[pt.surface_form].pct_rank if (pt.surface_form in clib_all_token_stats) else 100)
    if (rank > 80) and ((all_pos_tags[pt.pos_tag_id - 5]["name"]) == "N"):
        return True
    return False

def analyze_split(lib, ffi, test: str):
    global clib_all_token_stats
    success = True
    ret_pt = []
    num_sent = ffi.new("int[1]")
    use_mgt_machinery = int(0)
    sentences = lib.morphoy_parse_text(test.encode('utf-8'), num_sent, use_mgt_machinery)
    if num_sent[0] > 0:
        sent = sentences[0]
        for i in range(num_sent[0]):
            if not success:
                break
            if (sent.words_len > 0):
                word = sent.first_word[0]
                for j in range(sent.words_len):
                    # process word
                    pt = ParsedToken(word, ffi)
                    if is_rare_kinya_token(pt):
                        success = False
                        break
                    else:
                        ret_pt.append(pt)
                    if (j < (sent.words_len - 1)):
                        word = word.next[0]
            if (i < (num_sent[0] - 1)):
                sent = sent.next_sent[0]
    lib.morphoy_delete_sentences(sentences)
    rank = 100 * 10
    if success:
        rank = sum([(clib_all_token_stats[pt.surface_form].pct_rank if (pt.surface_form in clib_all_token_stats) else 100) for pt in ret_pt])
    return success,ret_pt,rank
def attempt_split_word(ffi, lib, pt: ParsedToken) -> List[ParsedToken]:
    global clib_all_token_stats
    tx = pt.raw_surface_form
    splits = [(tx[:i],tx[i:]) for i in range(1,len(tx))]
    results = [([pt],100*100)]
    for a_left,b_right in splits:
        left_lower = a_left.lower()
        right_lower = b_right.lower()
        if ((left_lower[-1] == 'i') or (left_lower[-1] == 'u') or (left_lower[-1] == 'o') or (left_lower[-1] == 'a') or (left_lower[-1] == 'e')):
            test = a_left + " " + b_right
            success, ret_pt, rank = analyze_split(lib, ffi, test)
            if success and (len(ret_pt) > 0):
                results.append((ret_pt, rank))
        elif ((right_lower[-1] == 'i') or (right_lower[-1] == 'u') or (right_lower[-1] == 'o') or (right_lower[-1] == 'a') or (right_lower[-1] == 'e')):
            test = a_left + "' " + b_right
            success, ret_pt, rank = analyze_split(lib, ffi, test)
            if success and (len(ret_pt) > 0):
                results.append((ret_pt, rank))
            if (left_lower == 'n'):
                test = left_lower + "i " + right_lower
                success, ret_pt, rank = analyze_split(lib, ffi, test)
                if success and (len(ret_pt) > 0):
                    results.append((ret_pt, rank))
            test = left_lower + "a " + right_lower
            success, ret_pt, rank = analyze_split(lib, ffi, test)
            if success and (len(ret_pt) > 0):
                results.append((ret_pt, rank))
    return min(results, key=lambda x:x[1])[0]
def morphoy_parse_input_text(ffi, lib, txt: str, attempt_auto_correction=False) -> (List[List[ParsedToken]], List[ParsedToken]):
    global clib_all_token_stats
    if clib_all_token_stats is None:
        clib_all_token_stats = get_all_token_stats()
    ret_sentences = []
    ret_words = []
    num_sent = ffi.new("int[1]")
    use_mgt_machinery = int(1)
    sentences = lib.morphoy_parse_text(txt.encode('utf-8'), num_sent, use_mgt_machinery)
    if num_sent[0] > 0:
        sent = sentences[0]
        for i in range(num_sent[0]):
            # New sentence
            mSentence = []
            if (sent.words_len > 0):
                word = sent.first_word[0]
                for j in range(sent.words_len):
                    # process word
                    pt = ParsedToken(word, ffi)
                    if is_rare_kinya_token(pt) and attempt_auto_correction:
                        pts = attempt_split_word(ffi, lib, pt)
                        for pptt in pts:
                            mSentence.append(pptt)
                            ret_words.append(pptt)
                    else:
                        mSentence.append(pt)
                        ret_words.append(pt)
                    if(j < (sent.words_len-1)):
                        word = word.next[0]
            ret_sentences.append(mSentence)
            if (i < (num_sent[0]-1)):
                sent = sent.next_sent[0]
    lib.morphoy_delete_sentences(sentences)
    return ret_sentences, ret_words

def parse_text_to_morpho_sentence(ffi, lib, txt: str, attempt_auto_correction=False) -> ParsedMorphoSentence:
    sentences, tokens = morphoy_parse_input_text(ffi, lib, txt, attempt_auto_correction=attempt_auto_correction)
    return ParsedMorphoSentence('_', parsed_tokens = tokens)
