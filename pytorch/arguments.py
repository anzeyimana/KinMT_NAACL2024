from __future__ import print_function, division

from argparse import ArgumentParser

from misc_functions import str2bool

def py_trainer_args(list_args=None, silent=False):
    parser = ArgumentParser(description="PyTorch Trainer")
    parser.add_argument("--morpho-dim-hidden", type=int, default=128)
    parser.add_argument("--stem-dim-hidden", type=int, default=256)

    parser.add_argument("--morpho-max-token-len", type=int, default=24)
    parser.add_argument("--morpho-rel-pos-bins", type=int, default=12)
    parser.add_argument("--morpho-max-rel-pos", type=int, default=12)

    parser.add_argument("--main-sequence-encoder-max-seq-len", type=int, default=512)
    parser.add_argument("--main-sequence-encoder-rel-pos-bins", type=int, default=256)
    parser.add_argument("--main-sequence-encoder-max-rel-pos", type=int, default=256)
    parser.add_argument("--dataset-max-seq-len", type=int, default=512)

    parser.add_argument("--morpho-dim-ffn", type=int, default=512)
    parser.add_argument("--main-sequence-encoder-dim-ffn", type=int, default=3072)

    parser.add_argument("--morpho-num-heads", type=int, default=4)
    parser.add_argument("--main-sequence-encoder-num-heads", type=int, default=12)

    parser.add_argument("--morpho-num-layers", type=int, default=4)
    parser.add_argument("--main-sequence-encoder-num-layers", type=int, default=12)

    parser.add_argument("--ft-reinit-layers", type=int, default=0)
    parser.add_argument("--ft-cwgnc", type=float, default=0.0)

    parser.add_argument("--morpho-dropout", type=float, default=0.1)
    parser.add_argument("--main-sequence-encoder-dropout", type=float, default=0.1)

    parser.add_argument("--layernorm-epsilon", type=float, default=1e-6)
    parser.add_argument("--pooler-dropout", type=float, default=0.3)

    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument("--load-saved-model", type=str2bool, default=True)
    parser.add_argument("--home-path", type=str, default="/home/user/MORPHO/")
    parser.add_argument("--train-parsed-corpus",type=str, default="data/parsed_morpho_corpus_docs_clean_2023-06-06.txt")
    parser.add_argument("--train-unparsed-corpus",type=str, default="data/morpho_corpus_docs_clean_2022-12-26.txt")
    parser.add_argument("--dev-unparsed-corpus",type=str, default="data/dev_morpho_corpus_docs_clean_2022-12-31.txt")
    parser.add_argument("--morpho_conf", type=str, default="config_morpho.conf")

    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--bert-accumulation-steps", type=int, default=160)
    parser.add_argument("--bert-num-iters", type=int, default=200000)
    parser.add_argument("--bert-warmup-iters", type=int, default=2000)
    parser.add_argument("--bert-number-of-load-batches", type=int, default=16000)

    parser.add_argument("--gpt-batch-size", type=int, default=14)
    parser.add_argument("--gpt-accumulation-steps", type=int, default=36)
    parser.add_argument("--gpt-num-iters", type=int, default=320000)
    parser.add_argument("--gpt-warmup-iters", type=int, default=2000)
    parser.add_argument("--gpt-number-of-load-batches", type=int, default=18000)

    parser.add_argument("--kinmt-seq2seq-config", type=str, default="kin2en")

    parser.add_argument("--kinmt-model-name", type=str, default="no_name")
    parser.add_argument("--kinmt-batch-max-tokens", type=int, default=4096)
    parser.add_argument("--kinmt-accumulation-steps", type=int, default=8)
    parser.add_argument("--kinmt-warmup-steps", type=int, default=2000)
    parser.add_argument("--kinmt-num-train-epochs", type=int, default=40)
    parser.add_argument("--kinmt-lexical-multiplier", type=int, default=1)
    parser.add_argument("--kinmt-peak-lr", type=float, default=0.001)
    parser.add_argument("--kinmt-use-bert", type=str2bool, default=False)
    parser.add_argument("--kinmt-use-gpt", type=str2bool, default=False)
    parser.add_argument("--kinmt-use-eval-data", type=str2bool, default=False)
    parser.add_argument("--kinmt-use-names-data", type=str2bool, default=True)
    parser.add_argument("--kinmt-use-foreign-terms", type=str2bool, default=False)
    parser.add_argument("--kinmt-use-copy-loss", type=str2bool, default=False)
    parser.add_argument("--kinmt-eval-benchmark-name", type=str2bool, default=True)
    parser.add_argument("--use-cross-positional-attn-bias", type=str2bool, default=False)
    parser.add_argument("--kinmt-extra-train-data-key", type=str, default='')
    parser.add_argument("--kinmt-engl-parse-per-token", type=str2bool, default=False)
    parser.add_argument("--kinmt-bert-large", type=str2bool, default=False)

    parser.add_argument("--kinmt_en2kin_prob_cutoff", type=float, default=0.3)
    parser.add_argument("--kinmt_en2kin_affix_prob_cutoff", type=float, default=0.3)
    parser.add_argument("--kinmt_en2kin_affix_min_prob", type=float, default=0.3)
    parser.add_argument("--kinmt_en2kin_lprob_score_delta", type=float, default=2.0)

    parser.add_argument("--max-input-lines", type=int, default=99999999)

    parser.add_argument("--multi-task-weighting", type=str2bool, default=False)

    parser.add_argument("--head-trunk", type=str2bool, default=False)
    parser.add_argument("--encoder-fine-tune", type=str2bool, default=True)

    parser.add_argument("--peak-lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--stop-grad-norm", type=float, default=1.0)
    parser.add_argument("--stop-loss", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=0.01)

    parser.add_argument("--corpus-id", type=int, default=1)

    parser.add_argument("--post-mlm-epochs", type=int, default=0)
    parser.add_argument("--pretrained-model-file", type=str, default="mybert_final_2022-12-31_operated_full_base_2022-12-13.pt")

    if list_args is not None:
        args = parser.parse_args(list_args)
    else:
        args = parser.parse_args()

    args.world_size = args.gpus

    if not silent:
        print('Call arguments:\n', args)

    return args

# Improvements/Innovations
# 1. Keep using empirically proven MyBERT architecture, given AlphaBERT did not materialize
# 2. Kept number of parameters reasonable given the limited amount of text data
# 3. Used Affix BOW transfer from morpho with bias vector parameter to sum up affixes or use in case no affixes are provided
# 4. Used Pre-LN Transformer architecture
# 5. Used Sigmoid+BCE for multi-label prediction of affix probability
# 5. Using NVIDIA apex's FusedLayerNorm across transformer
# 6. Using DeepSpeed toolkit with Zero optimization for heterogeneous training.
# 7. Combining both AFS+ADR for prediction of morphology
# x 8. Using Uncertainty weighting of losses for multi-task predictions
# Used Gradient Vaccine for multi-task learning
