#!/bin/bash

BASE_DIR=./ted-talks
TOKENIZED_DIR=$BASE_DIR/tokenized/combined-subs-sq-en
BINARIZED_DIR=$BASE_DIR/binarized/combined-subs-sq-en

fairseq-preprocess \
    --source-lang sq \
    --target-lang en \
    --trainpref $TOKENIZED_DIR/train \
    --validpref $TOKENIZED_DIR/val \
    --testpref $TOKENIZED_DIR/test \
    --destdir $BINARIZED_DIR/8000-joined-sq-en/ \
    --workers 20 \
    --nwordssrc 8000 \
    --nwordstgt 8000 \
    --joined-dictionary

#TOKENIZED_SOURCE=$TOKENIZED_DIR/combined-subs-sq-en
#
#fairseq-preprocess \
#    --source-lang sq \
#    --target-lang en \
#    --trainpref $TOKENIZED_SOURCE/train \
#    --validpref $TOKENIZED_SOURCE/val \
#    --testpref $TOKENIZED_SOURCE/test \
#    --destdir $TOKENIZED_DIR/8000-joined-sq-en/ \
#    --workers 20 \
#    --nwordssrc 8000 \
#    --nwordstgt 8000 \
#    --joined-dictionary