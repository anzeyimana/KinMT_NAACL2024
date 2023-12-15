import transformers
from tokenizers import SentencePieceBPETokenizer
if __name__ == '__main__':
    tokenizer_path = "tokenizers/sentencepiece/spm_engl/engl.spm.model"
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
    tk_tokenizer = SentencePieceBPETokenizer()

    text_file = open('kinmt/parallel_data_2022/txt/english_corpus_2023-06-10_en.txt', 'r', encoding='utf-8')

    tk_tokenizer.train_from_iterator(text_file, vocab_size=32_000, min_frequency=2, show_progress=True, special_tokens=special_tokens)

    text_file.close()

    tk_tokenizer.save(tokenizer_path)
    # convert
    model_length = 512
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tk_tokenizer, model_max_length=model_length, special_tokens=special_tokens)
    tokenizer.bos_token = "<s>"
    tokenizer.bos_token_id = tk_tokenizer.token_to_id("<s>")
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = tk_tokenizer.token_to_id("<pad>")
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = tk_tokenizer.token_to_id("</s>")
    tokenizer.unk_token = "<unk>"
    tokenizer.unk_token_id = tk_tokenizer.token_to_id("<unk>")
    tokenizer.cls_token = "<cls>"
    tokenizer.cls_token_id = tk_tokenizer.token_to_id("<cls>")
    tokenizer.sep_token = "<sep>"
    tokenizer.sep_token_id = tk_tokenizer.token_to_id("<sep>")
    tokenizer.mask_token = "<mask>"
    tokenizer.mask_token_id = tk_tokenizer.token_to_id("<mask>")
    # and save for later!
    tokenizer.save_pretrained("tokenizers/sentencepiece/english/")
