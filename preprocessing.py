import logging

logging.basicConfig(level=logging.INFO)


import argparse
import os
import glob
import time
import csv
import unicodedata
import html
import re

from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm
from functools import partial

import pandas as pd
import numpy as np

logging.info(pd.__version__)


# functions for cleaning and filtering the data
def str_strip(lang, text):

    return str(text).strip()


def normalize_unicode(lang, text):
    """Return the normal form for the Unicode string"""

    if NORM_CODE is not 'NONE' and NORM_CODE in ['NFC', 'NFD', 'NFKC', 'NFKD']:
        text = unicodedata.normalize(NORM_CODE, text)
    if lang == "th":
        return text.replace(u'\x99', u'').replace(u'\x9c', u'')
    return text


def normalize_text(lang, text):
    """
    Normalize quotation marks and handle hashtag sequences
    """

    text = re.sub(r'(^|[^S\w])#([A-Za-z0-9_]+)', '\\1｟#\\2｠', text)
    text = text.replace('“', '"')\
        .replace('”', '"')\
        .replace("‘", "'")\
        .replace("’", "'")
    return text


def html_unescape(lang, text):
    """Decoding HTML symbols"""

    return html.unescape(text)


def normalize_thai_text(lang, text):
    """
        Remove redundant symbols of tones and vowels.
        and subsitute [“เ”, “เ”] with “แ”.
    """

    if lang == "th":
        return pythainlp.util.normalize(text)
    return text


def th_contain_escape_code(lang, text):
    """Return True if text contains the defined escapte codes"""
    charsets = [
        '\\x9e',
        '\\x95',
        '\\x94',
        '\\x93',
        '\\x90',
        '\\x91',
    ]

    if lang == "th":
        for char in charsets:
            if char in repr(text):
                return True
    return False


CLEANING_RULES = [
    str_strip,
    html_unescape,
    normalize_unicode,
    normalize_text,
    normalize_thai_text,
]

FILTERING_RULES = [
    th_contain_escape_code
]


def clean_data(csv_dir, unicode_norm = 'none'):

    csv_file_paths = glob.glob(os.path.join(csv_dir, '*.csv'))
    file_to_df = {}

    for csv_file_path in csv_file_paths:
        # A csv file is returned as a data frame (two-dimensional data structure with labeled axes).
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        csv_filename = Path(csv_file_path).stem

        # clean text in both languages
        for lang in ['th', 'en']:

            df[f'{lang}_text'] = df[f'{lang}_text'].apply(str)

            for rule in CLEANING_RULES:

                df[f'{lang}_text'] = df[f'{lang}_text'].apply(
                    lambda x: rule(lang, x))
            for rule in FILTERING_RULES:

                _rule = partial(rule, lang)

                df[f'{lang}_text_to_drop'] = df[f'{lang}_text'].apply(_rule)
                df = df.drop(df[df[f'{lang}_text_to_drop'] == True].index)

        df = df.drop(columns=['en_text_to_drop', 'th_text_to_drop'])
        file_to_df[csv_filename] = df

    return file_to_df


# clean the dataset
NORM_CODE = 'NFKC'
DATASET = 'toy-ende' # note: the full dataset is 'scb-mt-en-th-2020'
DATA_DIR = os.path.join(".", DATASET)

file_to_df = clean_data(DATA_DIR, NORM_CODE)

def merge_csv(out_directory, df_list):

    if not os.path.exists(out_directory):
        os.makedirs(out_directory, exist_ok=True)

    out_path = os.path.join(out_directory, 'en-th.merged.csv')

    merged_item_ids = []

    for dataset_name, df in df_list.items():

        for index, _ in df.iterrows():
            sentence_id = f'{index}:{dataset_name}'
            merged_item_ids.append(sentence_id)

    merged_en_texts = pd.concat([df.en_text for _, df in df_list.items()]).apply(
        lambda x: str(x).strip())
    merged_th_texts = pd.concat([df.th_text for _, df in df_list.items()]).apply(
        lambda x: str(x).strip())

    # identify if the text has no duplicate
    merged_en_texts_is_duplicated = merged_en_texts.duplicated(
        keep=False).tolist()
    merged_th_texts_is_duplicated = merged_th_texts.duplicated(
        keep=False).tolist()

    with open(out_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['sentence_id', 'en', 'th',
                         'is_en_uniq', 'is_th_uniq'])

        for index, sentence_id in tqdm(enumerate(merged_item_ids), total=len(merged_item_ids)):

            is_en_uniq = not merged_en_texts_is_duplicated[index]
            is_th_uniq = not merged_th_texts_is_duplicated[index]

            en, th = merged_en_texts.iloc[index].replace(
                '\n', ''), merged_th_texts.iloc[index].replace('\n', '')

            writer.writerow([sentence_id, en, th, is_en_uniq, is_th_uniq])


# merge data
out_dir = os.path.join(".", 'merged', DATASET)
merge_csv(out_dir, file_to_df)

# code for splitting the dataset

def print_sub_dataset_dist(series):
    """Helper function for printing number of sample sentences"""
    N = sum(series.values)
    for dataset, count in series.items():
        print(f'{dataset:25}: {count:8,} ( {float(count/N*100):5.2f}% )')


def split_dataset(path_merged_csv, out_dir, train_ratio, val_ratio, test_ratio, seed):
    """Split the given merged dataset(csv) in to train, val, test set.
        Output the split dataset(csv) in 'out_dir'.
    """
    df = pd.read_csv(path_merged_csv, encoding='utf-8', engine='python')
    df.is_en_uniq.astype(bool)
    df.is_th_uniq.astype(bool)

    df['dataset'] = df['sentence_id'].apply(lambda x: x.split(':')[-1])
    train_df, val_df, test_df = None, None, None

    N = df.shape[0]

    print('\nSummary: Number of segment pairs for each sub-dataset and percentage\n')

    print_sub_dataset_dist(df['dataset'].value_counts())

    print('')


    n_val = int(N * val_ratio)
    n_test = int(N * test_ratio)

    val_test_df = df[(df['is_en_uniq'] == True) & (
            df['is_th_uniq'] == True)].sample(n=n_val + n_test, random_state=seed)

    val_test_ids = val_test_df.sentence_id.tolist()

    val_df = val_test_df.sample(n=n_val, random_state=seed)
    val_ids = val_df.sentence_id.tolist()

    test_df = val_test_df[val_test_df['sentence_id'].isin( val_ids) == False]
    train_df = df[df['sentence_id'].isin(val_test_ids) == False]

    print('\nDone spliting train/val/test set')
    print( f'\nRatio (train, val, test): ({train_ratio:2}, {val_ratio:2}, {test_ratio:2})')
    print(f'Number of segment pairs (train, val, test): {train_df.shape[0]:6,} | {val_df.shape[0]:6,} | {test_df.shape[0]:6,}')



    if not os.path.exists(out_dir):
        print(f'\nCreate a directory at: `{out_dir}`')
        os.makedirs(out_dir, exist_ok=True)

    print(f'\n\nStart writing output files to `{out_dir}`')

    train_df = train_df.drop(columns=['dataset'])
    test_df = test_df.drop(columns=['dataset'])


    val_df = val_df.drop(columns=['dataset'])
    val_df.to_csv(os.path.join(
        out_dir, f'en-th.merged.val.csv'), encoding='utf-8')

    train_df.to_csv(os.path.join(
        out_dir, f'en-th.merged.train.csv'), encoding='utf-8')
    test_df.to_csv(os.path.join(
        out_dir, f'en-th.merged.test.csv'), encoding='utf-8')

    print('\nDone writing files.')

# run the code
path_merged_csv = os.path.join('dataset', 'merged', 'toy','en-th.merged.csv')
out_dir = os.path.join('dataset', 'split','toy')
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
seed = 2020
split_dataset(path_merged_csv, out_dir, train_ratio, val_ratio, test_ratio, seed)