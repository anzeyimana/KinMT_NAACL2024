python3 kinmt_en2kin_hparams_search.py \
  -g 1 \
  --load-saved-model=true \
  --kinmt-use-bert=true \
  --kinmt-use-gpt=false \
  --use-cross-positional-attn-bias=true \
  --kinmt-use-copy-loss=false \
  --kinmt-engl-parse-per-token=false \
  --kinmt-model-name="en2kin_base_back_trans_eval_bert_xpos_2023-06-08.pt_ep1_121K"
