#!/bin/bash

BASE_DIR=./ted-talks
TOKENIZED_DIR=$BASE_DIR/tokenized
TOKENIZED_SOURCE=$TOKENIZED_DIR/combined-subs-en-sq

fairseq-preprocess \
    --source-lang sq \
    --target-lang en \
    --trainpref $TOKENIZED_SOURCE/train \
    --validpref $TOKENIZED_SOURCE/val \
    --testpref $TOKENIZED_SOURCE/test \
    --destdir $TOKENIZED_DIR/8000-joined/ \
    --workers 20 \
    --nwordssrc 8000 \
    --nwordstgt 8000 \
    --joined-dictionary