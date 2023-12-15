import numpy as np
from datasets import load_metric
from misc_functions import time_now, read_lines
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets.arrow_dataset import Dataset

tokenizer = None
model = None
metric = None

max_input_length = 128
max_target_length = 128

def kin2en_model_setup(src_lang='rw', tgt_lang='en'):
    global tokenizer
    global model
    global metric
    model_checkpoint = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    metric = load_metric("sacrebleu")
    return tokenizer, model, metric

def kin2en_dataset_preprocess(examples):
    global tokenizer
    global model
    global metric
    inputs = examples['rw']
    targets = examples['en']
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    global tokenizer
    global model
    global metric
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    global tokenizer
    global model
    global metric
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def kin2en_train():
    global tokenizer
    global model
    global metric
    src_lang = 'rw'
    tgt_lang = 'en'
    kin2en_model_setup(src_lang=src_lang, tgt_lang=tgt_lang)

    # Read data
    train_items = ['jw', 'gazette',
              'habumuremyi_combined', 'kinyarwandanet',
              'stage1', 'stage2', 'stage3', 'stage3_residuals',
               'numeric_examples', 'names_and_numbers', 'foreign_terms']
    dev_items = ['flores200_dev', 'mafand_mt_dev', 'tico19_dev']

    training_set = []
    validation_set = []

    print(time_now(), 'Reading training dataset ...', flush=True)
    for item in train_items:
        rw_lines = read_lines(f'kinmt/parallel_data_2022/txt/{item}_rw.txt')
        en_lines = read_lines(f'kinmt/parallel_data_2022/txt/{item}_en.txt')
        pairs = [(rw, en) for rw, en in zip(rw_lines, en_lines) if ((len(rw.split()) < 300) and (len(en.split()) < 300))]
        training_set.extend(pairs)
    training_dataset = Dataset.from_dict({'rw':[rw for (rw,en) in training_set],'en':[en for (rw,en) in training_set]})
    tokenized_training_dataset = training_dataset.map(kin2en_dataset_preprocess, batched=True)
    print(time_now(), f'Read {len(training_set)} training examples!', flush=True)

    print(time_now(), 'Reading validation dataset ...', flush=True)
    for item in dev_items:
        rw_lines = read_lines(f'kinmt/parallel_data_2022/txt/{item}_rw.txt')
        en_lines = read_lines(f'kinmt/parallel_data_2022/txt/{item}_en.txt')
        pairs = [(rw, en) for rw, en in zip(rw_lines, en_lines) if ((len(rw.split()) < 300) and (len(en.split()) < 300))]
        validation_set.extend(pairs)
    validation_dataset = Dataset.from_dict({'rw':[rw for (rw,en) in validation_set],'en':[en for (rw,en) in validation_set]})
    tokenized_validation_dataset = validation_dataset.map(kin2en_dataset_preprocess, batched=True)
    print(time_now(), f'Read {len(validation_set)} validation examples!', flush=True)

    model_checkpoint = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    batch_size = 16
    model_name = model_checkpoint.split("/")[-1]
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-{src_lang}-to-{tgt_lang}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=4,
        num_train_epochs=24,
        predict_with_generate=True
    )
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_training_dataset,
        eval_dataset=tokenized_validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

if __name__ == '__main__':
    import os
    os.environ["WANDB_DISABLED"] = "true"
    kin2en_train()
