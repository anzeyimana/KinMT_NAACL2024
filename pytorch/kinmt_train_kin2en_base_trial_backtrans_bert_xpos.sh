python3 kin2en_no_amp_trainer.py \
  -g 8 \
  --load-saved-model=false \
  --kinmt-batch-max-tokens=4096 \
  --kinmt-accumulation-steps=8 \
  --kinmt-use-bert=true \
  --kinmt-use-gpt=false \
  --use-cross-positional-attn-bias=true \
  --kinmt-use-copy-loss=false \
  --kinmt-use-names-data=true \
  --kinmt-use-foreign-terms=true \
  --kinmt-use-eval-data=false \
  --kinmt-num-train-epochs=64 \
  --kinmt-lexical-multiplier=1 \
  --kinmt-peak-lr=0.001 \
  --kinmt-warmup-steps=16000 \
  --kinmt-bert-large=true \
  --kinmt-extra-train-data-key="english_clean_corpus_2023-07-31" \
  --kinmt-model-name="kin2en_base_large_bert_trial_backtrans_eval_xpos_2023-08-07"
