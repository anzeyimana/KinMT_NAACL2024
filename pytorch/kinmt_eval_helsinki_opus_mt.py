from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def init_helsinki_opus_mt(src_lang='rw', tgt_lang='en'):
    tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}")
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}")
    return (tokenizer, mt_model)

def helsinki_opus_mt_translate(tokenizer_mt_model_setup, text:str) -> str:
    (tokenizer, mt_model) = tokenizer_mt_model_setup
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = mt_model.generate(input_ids, max_length=512)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation
