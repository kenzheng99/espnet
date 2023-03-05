#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/train_asr_tedlium_conformer_wavlm_last.yaml
bpe_train_text=dump/raw/train_sp/text
# lm_config=conf/train_lm.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                               \
    --lang en                                          \
    --audio_format flac                                \
    --feats_type raw                                   \
    --token_type bpe                                   \
    --nbpe 500                                         \
    --feats_normalize utterance_mvn                    \
    --bpe_train_text ${bpe_train_text}                 \
    --use_lm ${use_lm}                                 \
    --asr_config "${asr_config}"                       \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --ngpu 4                                           \
    --stage 11                                         \
