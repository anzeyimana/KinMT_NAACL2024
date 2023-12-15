python3 kin2en_trainer.py \
  -g 4 \
  --load-saved-model=false \
  --kinmt-batch-max-tokens=2048 \
  --kinmt-accumulation-steps=16 \
  --kinmt-use-bert=false \
  --kinmt-use-gpt=false \
  --use-cross-positional-attn-bias=false \
  --kinmt-use-names-data=false \
  --kinmt-model-name="kin2en_base_no_names_2023-04-27"
