import logging
import argparse
import os
import glob
import time
import csv
import unicodedata
import html
import re
import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm
from functools import partial


logging.basicConfig(level=logging.INFO)
logging.info(pd.__version__)

GERMAN_LANGUAGE = "de"
ENGLISH_LANGUAGE = "en"


# functions for cleaning and filtering the data
def str_strip(lang, text):
    return str(text).strip()


def normalize_unicode(lang, text):
    """Return the normal form for the Unicode string"""

    if NORM_CODE is not 'NONE' and NORM_CODE in ['NFC', 'NFD', 'NFKC', 'NFKD']:
        text = unicodedata.normalize(NORM_CODE, text)
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


CLEANING_RULES = [
    str_strip,
    html_unescape,
    normalize_unicode,
    normalize_text,
]


def clean_data(csv_dir, unicode_norm='none'):

    csv_file_paths = glob.glob(os.path.join(csv_dir, '*.tsv'))
    file_to_df_dictionary = {}

    for csv_file_path in csv_file_paths:
        # A csv file is returned as a data frame (two-dimensional data structure with labeled axes).
        df = pd.read_csv(
            csv_file_path,
            encoding='utf-8',
            sep="\t",
            names=[f"{GERMAN_LANGUAGE}_text", f"{ENGLISH_LANGUAGE}_text"]
        )
        csv_filename = Path(csv_file_path).stem

        # clean text in both languages
        for lang in [GERMAN_LANGUAGE, ENGLISH_LANGUAGE]:

            df[f'{lang}_text'] = df[f'{lang}_text'].apply(str)

            for rule in CLEANING_RULES:
                df[f'{lang}_text'] = df[f'{lang}_text'].apply(
                    lambda x: rule(lang, x)
                )

        file_to_df_dictionary[csv_filename] = df

    return file_to_df_dictionary


# clean the dataset
NORM_CODE = 'NFKC'
DATASET = "toy-ende"  # note: the full dataset is 'scb-mt-en-th-2020'
DATA_DIR = os.path.join(".", DATASET)

file_to_df = clean_data(DATA_DIR, NORM_CODE)


def merge_csv(out_directory, df_list):
    out_path = Path.joinpath(out_directory, "en-de.merged.csv")

    merged_item_ids = []

    for dataset_name, df in df_list.items():

        for index, _ in df.iterrows():
            sentence_id = f'{index}:{dataset_name}'
            merged_item_ids.append(sentence_id)

    merged_en_texts = pd.concat([df.en_text for _, df in df_list.items()]).apply(
        lambda x: str(x).strip()
    )
    merged_de_texts = pd.concat([df.de_text for _, df in df_list.items()]).apply(
        lambda x: str(x).strip()
    )

    # identify if the text has no duplicate
    merged_en_texts_is_duplicated = merged_en_texts.duplicated(
        keep=False
    ).tolist()
    merged_de_texts_is_duplicated = merged_de_texts.duplicated(
        keep=False
    ).tolist()

    with open(out_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(
            f, delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([
            'sentence_id', 'en', 'de',
            'is_en_uniq', 'is_de_uniq'
        ])

        for index, sentence_id in tqdm(enumerate(merged_item_ids), total=len(merged_item_ids)):

            is_en_uniq = not merged_en_texts_is_duplicated[index]
            is_de_uniq = not merged_de_texts_is_duplicated[index]

            en, de = merged_en_texts.iloc[index].replace(
                '\n', ''), merged_de_texts.iloc[index].replace('\n', '')

            writer.writerow([sentence_id, en, de, is_en_uniq, is_de_uniq])


# merge data
cwd = Path.cwd()
out_dir = Path.joinpath(cwd, "merged", DATASET)
Path.mkdir(out_dir, parents=True, exist_ok=True)
merge_csv(out_dir, file_to_df)


# code for splitting the dataset
def print_sub_dataset_dist(series):
    """Helper function for printing number of sample sentences"""
    N = sum(series.values)
    for dataset, count in series.items():
        print(f'{dataset:25}: {count:8,} ( {float(count/N*100):5.2f}% )')


def split_dataset(path_merged_csv, out_dir, train_ratio, val_ratio, test_ratio, seed):
    """
    Split the given merged dataset(csv) in to train, val, test set.
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
path_merged_csv = os.path.join('dataset', 'merged', 'toy', 'en-de.merged.csv')
out_dir = os.path.join('dataset', 'split', 'toy')
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
seed = 2020
split_dataset(path_merged_csv, out_dir, train_ratio, val_ratio, test_ratio, seed)