#!/bin/bash

BASE_DIR=./ted-talks
TOKENIZED_DIR=$BASE_DIR/tokenized
BINARIZED_DIR=$TOKENIZED_DIR/8000-joined/

GPU_ID=0
OUT_DIR='ted-talks/8000-joined'
MAX_TOKEN=9750
MAX_EPOCH=15


CUDA_VISIBLE_DEVICES=0
fairseq-train $BINARIZED_DIR \
    --cpu \
    --arch transformer \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
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
    --fp16 \
    --no-epoch-checkpoints \
    --max-epoch $MAX_EPOCH \
    --num-workers 0