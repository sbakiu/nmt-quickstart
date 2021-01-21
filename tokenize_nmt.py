# code for training SentencePiece and tokenize the data
import glob
import shutil
import uuid
from pathlib import Path

import pandas as pd

import sentencepiece as spm
from functools import partial
import os


def write_txt(path, sentences):
    """Write the given sentences into the given path"""
    with open(path, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(str(sent) + '\n')

def train_spm(train_dataset_path, side, vocab_size, spm_out_dir, lower):
    """Train a SentencePiece model for BPE tokenization.

    train_dataset_path: path to training data in csv
    side: 'th' or 'en'
    vocab_size: vocabulary size
    spm_out_dir: output directory to store the trained model
    lower: if true, lower case the text
    """

    if not os.path.exists('temp'):
        print('\nCreate a temp directory')
        os.makedirs('temp', exist_ok=True)

    df = pd.read_csv(train_dataset_path, encoding='utf-8')

    if lower:
        df[side] = df[side].apply(lambda x: x.lower())

    sentences = df[side].tolist()

    train_dataset_path = Path(train_dataset_path)
    current_dir = train_dataset_path.cwd()

    case = 'uncased' if lower else 'cased'
    spm_file_prefix = f'spm.{side}.v-{vocab_size}.{case}'

    if not os.path.exists('spm_out_dir'):
        print(f'\nCreate a directory at `{spm_out_dir}`.')
        os.makedirs(spm_out_dir, exist_ok=True)

    spm_model_path = os.path.join(spm_out_dir, f'{spm_file_prefix}.model')

    if not os.path.exists(spm_model_path):

        temp_filepath = f'./temp/temp.sentences.{uuid.uuid1()}.{side}'

        print(f'\nWrite sentences to a temporary file at : {temp_filepath}')
        write_txt(temp_filepath, sentences)

        print('\nSentencePiece model is not found.')
        print(
            f'\nBegin training SentencePiece model, filename: {spm_file_prefix}.model')

        spm.SentencePieceTrainer.Train(
            f'--input={temp_filepath} --character_coverage=1.0 --model_prefix={spm_file_prefix} --vocab_size={vocab_size}')

        spm_model_path = f'./{spm_file_prefix}.model'
        spm_vocab_path = f'./{spm_file_prefix}.vocab'

        print(f'Move vocab and model files to {spm_out_dir}')
        spm_model_path = shutil.move(spm_model_path, os.path.join(
            spm_out_dir, f'{spm_file_prefix}.model'))
        spm_vocab_path = shutil.move(spm_vocab_path, os.path.join(
            spm_out_dir, f'{spm_file_prefix}.vocab'))

        print('\nDone')

        # remove tempfile
        print('\nRemove temp file')
        os.remove(temp_filepath)
    else:
        spm_model_path = os.path.join(spm_out_dir, f'{spm_file_prefix}.model')
        spm_vocab_path = os.path.join(spm_out_dir, f'{spm_file_prefix}.vocab')
        print('SentencePiece model was found.')

    print(f'Begin loading SentencePiece model from {spm_model_path}')
    model = spm.SentencePieceProcessor()

    print(f'Done loading SPM model')

    model.Load(spm_model_path)

    return model


def spm_tokenize(split_dataset_dir,
                 src_lang, tgt_lang,
                 src_uncased, tgt_uncased,
                 src_spm_vocab_size, tgt_spm_vocab_size,
                 out_dir, spm_out_dir):
    """Tokenize the dataset in 'split_dataset_dir' and output the result in 'spm_out_dir'.

        src_lang, tgt_lang: 'th' or 'en'
        src_uncased, tgt_uncased: if true, lowercasing the text
        src_spm_vocab_size, tgt_spm_vocab_size: vocabulary size
        out_dir: output dir that store the tokenized dataset
        spm_out_dir: output dir of the trained SentencePiece model
    """

    file_paths = glob.glob(os.path.join(split_dataset_dir, '*.csv'))
    file_paths = list(filter(lambda x: 'train' in x, file_paths))

    assert len(file_paths) == 1
    train_filepath = file_paths[0]

    # Train SentencePiece model on train set for src_lang

    src_spm_model = train_spm(
        train_filepath, src_lang, src_spm_vocab_size, spm_out_dir, lower=src_uncased)

    _src_tokenizer = partial(src_spm_model.EncodeAsPieces)

    # Train SentencePiece model on train set for tgt_lang

    tgt_spm_model = train_spm(
        train_filepath, tgt_lang, tgt_spm_vocab_size, spm_out_dir, lower=tgt_uncased)

    _tgt_tokenizer = partial(tgt_spm_model.EncodeAsPieces)

    file_paths = glob.glob(os.path.join(split_dataset_dir, '*.csv'))
    for file_path in file_paths:

        lang_pair, name, split = Path(file_path).stem.split('.')

        df = pd.read_csv(file_path, encoding='utf-8')

        df[src_lang] = df[src_lang].apply(str)
        df[tgt_lang] = df[tgt_lang].apply(str)

        if src_uncased:
            df[src_lang] = df[src_lang].apply(lambda x: x.lower())
        if tgt_uncased:
            df[src_lang] = df[src_lang].apply(lambda x: x.lower())

        src_tokens = df[src_lang].apply(_src_tokenizer)
        tgt_tokens = df[tgt_lang].apply(_tgt_tokenizer)

        if not os.path.exists(out_dir):
            print(f'Create a directiony at `{out_dir}`.')
            os.makedirs(out_dir, exist_ok=True)

        src_out_path = os.path.join(out_dir, f'{split}.{src_lang}')
        tgt_out_path = os.path.join(out_dir, f'{split}.{tgt_lang}')

        print(
            f'\n - Write tokenized result of {src_lang} langauge of the {split} set, to {src_out_path}')

        write_tokenized_result(src_tokens, src_out_path)

        print(
            f'\n - Write tokenized result of {tgt_lang} langauge of the {split} set, to {tgt_out_path}')

        write_tokenized_result(tgt_tokens, tgt_out_path)

    print('\n\nDone.')


def write_tokenized_result(series, path):
    """Write tokenized texts stored as Pandas series in to a given file path"""

    with open(path, 'w', encoding='utf-8') as f:
        for tokens in series.tolist():
            line = ' '.join(tokens)
            f.write(f"{line}\n")


DATASET = "toy-ende" # note: the full dataset is 'scb-mt-en-th-2020'
DATA_DIR = os.path.join(".", DATASET)

split_dataset_dir = os.path.join('dataset', 'split', DATASET)

src_lang = "th"
tgt_lang = "en"

src_uncased =  True
tgt_uncased  = True

src_spm_vocab_size  = 10000
tgt_spm_vocab_size  = 10000

out_dir = os.path.join('dataset', 'tokenized', DATASET)
spm_out_dir = os.path.join('dataset', 'spm', DATASET)

spm_tokenize(split_dataset_dir,
             src_lang, tgt_lang,
             src_uncased, tgt_uncased,
             src_spm_vocab_size,tgt_spm_vocab_size,
             out_dir, spm_out_dir)