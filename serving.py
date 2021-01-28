import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from fairseq.models.transformer import TransformerModel
sq2en = TransformerModel.from_pretrained(
    './checkpoints-sq-en/ted-talks/8000-joined',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='ted-talks/tokenized/8000-joined-en-sq',
    # bpe='subword_nmt',
    # bpe_codes='data-bin/wmt17_zh_en_full/zh.code'
)

translate = sq2en.translate(["Hello", "t"])
logging.info(f"T: {translate}")
