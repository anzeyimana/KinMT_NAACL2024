python3 kin2en_ddp_trainer.py \
  -g 8 \
  --load-saved-model=false \
  --kinmt-batch-max-tokens=2048 \
  --kinmt-accumulation-steps=16 \
  --kinmt-use-bert=true \
  --kinmt-use-gpt=false \
  --use-cross-positional-attn-bias=false \
  --kinmt-use-copy-loss=false \
  --kinmt-use-names-data=true \
  --kinmt-model-name="kin2en_base_bert_2023-04-30"
