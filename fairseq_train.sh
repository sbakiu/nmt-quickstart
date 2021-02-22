#!/bin/bash

BASE_DIR=./ted-talks
BINARIZED_DIR=$BASE_DIR/binarized/combined-subs-sq-en/8000-joined-sq-en

OUT_DIR='ted-talks/trained-8000-joined-sq-en'
MAX_TOKEN=9750
MAX_EPOCH=10

fairseq-train $BINARIZED_DIR \
    --cpu \
    --arch transformer \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --dropout 0.3 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens $MAX_TOKEN \
    --save-dir ./checkpoints/$OUT_DIR \
    --tensorboard-logdir ./checkpoints/$OUT_DIR/log \
    --update-freq 16 \
    --no-epoch-checkpoints \
    --max-epoch $MAX_EPOCH \
    --num-workers 0