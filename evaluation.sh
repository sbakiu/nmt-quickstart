# shellcheck disable=SC2002
cat ./ted-talks/tokenized/combined-subs-sq-en/test.sq \
    | fairseq-interactive ./ted-talks/binarized/combined-subs-sq-en/8000-joined-sq-en \
      --task translation \
      --source-lang sq \
      --target-lang en \
      --path ./checkpoints/ted-talks/trained-8000-joined-sq-en/checkpoint_best.pt \
      --buffer-size 2500 \
      --max-tokens 2000 \
      --beam 4 \
    > ./hypo.sq-en

#grep ^H ./hypo.sq-en \
#     | cut -f3 \
#     > ./hypo.sq-en.sys
