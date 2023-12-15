python3 kinmt_seq2seq_eval_benchmark.py \
  --load-saved-model=false \
  --kinmt-batch-max-tokens=4096 \
  --kinmt-accumulation-steps=8 \
  --kinmt-use-bert=false \
  --kinmt-use-gpt=false \
  --use-cross-positional-attn-bias=true \
  --kinmt-use-copy-loss=false \
  --kinmt-use-names-data=true \
  --kinmt-use-foreign-terms=false \
  --kinmt-use-eval-data=false \
  --kinmt-num-train-epochs=24 \
  --kinmt-peak-lr=0.0008 \
  --kinmt-seq2seq-config="en2kin" \
  --kinmt-model-name="seq2seq_en2kin_base_eval_no_bert_xpos_2023-06-26.pt_best_valid_loss"

python3 kinmt_seq2seq_eval_benchmark.py \
  --load-saved-model=false \
  --kinmt-batch-max-tokens=4096 \
  --kinmt-accumulation-steps=8 \
  --kinmt-use-bert=false \
  --kinmt-use-gpt=false \
  --use-cross-positional-attn-bias=true \
  --kinmt-use-copy-loss=false \
  --kinmt-use-names-data=true \
  --kinmt-use-foreign-terms=false \
  --kinmt-use-eval-data=false \
  --kinmt-num-train-epochs=24 \
  --kinmt-peak-lr=0.0008 \
  --kinmt-seq2seq-config="en2kin" \
  --kinmt-model-name="seq2seq_en2kin_base_eval_no_bert_xpos_2023-06-26"

# ----------------------------------------------------------------------------------------------------------------------
python3 kinmt_seq2seq_eval_benchmark.py \
  --load-saved-model=false \
  --kinmt-batch-max-tokens=4096 \
  --kinmt-accumulation-steps=8 \
  --kinmt-use-bert=false \
  --kinmt-use-gpt=false \
  --use-cross-positional-attn-bias=true \
  --kinmt-use-copy-loss=false \
  --kinmt-use-names-data=true \
  --kinmt-use-foreign-terms=true \
  --kinmt-use-eval-data=false \
  --kinmt-num-train-epochs=24 \
  --kinmt-peak-lr=0.0008 \
  --kinmt-seq2seq-config="kin2en" \
  --kinmt-model-name="seq2seq_kin2en_base_eval_no_bert_xpos_2023-06-26.pt_best_valid_loss"

python3 kinmt_seq2seq_eval_benchmark.py \
  --load-saved-model=false \
  --kinmt-batch-max-tokens=4096 \
  --kinmt-accumulation-steps=8 \
  --kinmt-use-bert=false \
  --kinmt-use-gpt=false \
  --use-cross-positional-attn-bias=true \
  --kinmt-use-copy-loss=false \
  --kinmt-use-names-data=true \
  --kinmt-use-foreign-terms=true \
  --kinmt-use-eval-data=false \
  --kinmt-num-train-epochs=24 \
  --kinmt-peak-lr=0.0008 \
  --kinmt-seq2seq-config="kin2en" \
  --kinmt-model-name="seq2seq_kin2en_base_eval_no_bert_xpos_2023-06-26"
