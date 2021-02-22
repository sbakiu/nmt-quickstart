import logging
from fairseq.models.transformer import TransformerModel

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

en2sq = TransformerModel.from_pretrained(
    './checkpoints/ted-talks/trained-8000-joined-en-sq',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='ted-talks/tokenized/8000-joined-en-sq',
    bpe="sentencepiece",  # bpe='subword_nmt',
    sentencepiece_model="ted-talks/spm/combined-subs-en-sq/spm.en.v-1000.uncased.model" # bpe_codes='data-bin/wmt17_zh_en_full/zh.code'
)

translate = en2sq.translate(["dog"])
logging.info(f"T: {translate}")
