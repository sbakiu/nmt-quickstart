import logging
import os
import glob
import csv
import unicodedata
import html
import re
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm


logging.basicConfig(
    format="%(asctime)s %(levelname)-4s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


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


def clean_data(csv_dir, lang1, lang2):

    path = Path.joinpath(csv_dir, '*.tsv')
    csv_file_paths = glob.glob(str(path))
    file_to_df_dictionary = {}

    for csv_file_path in csv_file_paths:
        # A csv file is returned as a data frame (two-dimensional data structure with labeled axes).
        df = pd.read_csv(
            csv_file_path,
            encoding='utf-8',
            sep="\t"
        )
        csv_filename = Path(csv_file_path).stem

        # clean text in both languages
        for lang in [lang1, lang2]:

            df[f'{lang}'] = df[f'{lang}'].apply(str)

            for rule in CLEANING_RULES:
                df[f'{lang}'] = df[f'{lang}'].apply(
                    lambda x: rule(lang, x)
                )

        file_to_df_dictionary[csv_filename] = df

    return file_to_df_dictionary


def merge_csv(out_directory, df_list, lang1, lang2):
    logging.info("Merging CSV files")
    out_path = Path.joinpath(out_directory, f"{ALBANIAN_LANG_ISO_CODE}-{ENGLISH_LANG_ISO_CODE}.merged.csv")

    merged_item_ids = []

    for dataset_name, df in df_list.items():

        for index, _ in df.iterrows():
            sentence_id = f'{index}:{dataset_name}'
            merged_item_ids.append(sentence_id)

    merged_lang1_texts = pd.concat([df[lang1] for _, df in df_list.items()]).apply(
        lambda x: str(x).strip()
    )
    merged_lang2_texts = pd.concat([df[lang2] for _, df in df_list.items()]).apply(
        lambda x: str(x).strip()
    )

    # identify if the text has no duplicate
    merged_lang1_texts_is_duplicated = merged_lang1_texts.duplicated(
        keep=False
    ).tolist()
    merged_lang2_texts_is_duplicated = merged_lang2_texts.duplicated(
        keep=False
    ).tolist()

    with open(out_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(
            f,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([
            'sentence_id',
            lang1,
            lang2,
            f"is_{lang1}_uniq",
            f"is_{lang2}_uniq"
        ])

        for index, sentence_id in tqdm(enumerate(merged_item_ids), total=len(merged_item_ids)):

            is_lang1_uniq = not merged_lang1_texts_is_duplicated[index]
            is_lang2_uniq = not merged_lang2_texts_is_duplicated[index]

            merged_text_lang1 = merged_lang1_texts.iloc[index].replace('\n', '')
            merged_text_lang2 = merged_lang2_texts.iloc[index].replace('\n', '')

            writer.writerow([sentence_id, merged_text_lang1, merged_text_lang2, is_lang1_uniq, is_lang2_uniq])


# code for splitting the dataset
def print_sub_dataset_dist(series):
    """Helper function for printing number of sample sentences"""
    N = sum(series.values)
    for dataset, count in series.items():
        print(f'{dataset:25}: {count:8,} ( {float(count/N*100):5.2f}% )')


def split_dataset(path_merged_csv, out_dir, train_ratio, val_ratio, test_ratio, seed, lang1, lang2):
    """
    Split the given merged dataset(csv) in to train, val, test set.
    Output the split dataset(csv) in 'out_dir'.
    """
    df = pd.read_csv(path_merged_csv, encoding='utf-8', engine='python')
    df[f"is_{lang1}_uniq"].astype(bool)
    df[f"is_{lang2}_uniq"].astype(bool)

    df['dataset'] = df['sentence_id'].apply(lambda x: x.split(':')[-1])
    train_df, val_df, test_df = None, None, None

    N = df.shape[0]

    print('\nSummary: Number of segment pairs for each sub-dataset and percentage\n')

    print_sub_dataset_dist(df['dataset'].value_counts())

    print('')

    n_val = int(N * val_ratio)
    n_test = int(N * test_ratio)

    val_test_df = df[
        (df[f"is_{lang1}_uniq"] == True) & (df[f"is_{lang2}_uniq"] == True)
    ].sample(n=n_val + n_test, random_state=seed)

    val_test_ids = val_test_df.sentence_id.tolist()

    val_df = val_test_df.sample(n=n_val, random_state=seed)
    val_ids = val_df.sentence_id.tolist()

    test_df = val_test_df[val_test_df['sentence_id'].isin(val_ids) == False]
    train_df = df[df['sentence_id'].isin(val_test_ids) == False]

    print('\nDone spliting train/val/test set')
    print(f'\nRatio (train, val, test): ({train_ratio:2}, {val_ratio:2}, {test_ratio:2})')
    print(
        f'Number of segment pairs (train, val, test): '
        f'{train_df.shape[0]:6,} | {val_df.shape[0]:6,} | {test_df.shape[0]:6,}'
    )

    if not os.path.exists(out_dir):
        print(f'\nCreate a directory at: `{out_dir}`')
        os.makedirs(out_dir, exist_ok=True)

    print(f'\n\nStart writing output files to `{out_dir}`')

    train_df = train_df.drop(columns=['dataset'])
    test_df = test_df.drop(columns=['dataset'])

    val_df = val_df.drop(columns=['dataset'])
    val_df.to_csv(
        os.path.join(out_dir, f"{lang1}-{lang2}.merged.val.csv"),
        encoding='utf-8'
    )

    train_df.to_csv(
        os.path.join(out_dir, f"{lang1}-{lang2}.merged.train.csv"),
        encoding='utf-8'
    )
    test_df.to_csv(
        os.path.join(out_dir, f"{lang1}-{lang2}.merged.test.csv"),
        encoding='utf-8'
    )

    print('\nDone writing files.')


GERMAN_LANG_ISO_CODE = "de"
ENGLISH_LANG_ISO_CODE = "en"
ALBANIAN_LANG_ISO_CODE = "sq"

# clean the dataset
NORM_CODE = 'NFKC'
BASE_DIR = "ted-talks"
DATASET = "combined-subs-en-sq"

cwd = Path.cwd()
base_path = Path.joinpath(cwd, BASE_DIR)
DATA_DIR = Path.joinpath(base_path, DATASET)

file_to_df = clean_data(DATA_DIR, ALBANIAN_LANG_ISO_CODE, ENGLISH_LANG_ISO_CODE)

out_dir = Path.joinpath(base_path, "merged", DATASET)
Path.mkdir(out_dir, parents=True, exist_ok=True)

merge_csv(out_dir, file_to_df, ALBANIAN_LANG_ISO_CODE, ENGLISH_LANG_ISO_CODE)

path_merged_csv = Path.joinpath(
    base_path,
    "merged",
    DATASET,
    f"{ALBANIAN_LANG_ISO_CODE}-{ENGLISH_LANG_ISO_CODE}.merged.csv"
)

out_dir = Path.joinpath(base_path, "split", DATASET)
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
seed = 2021
split_dataset(
    path_merged_csv,
    out_dir,
    train_ratio,
    val_ratio,
    test_ratio,
    seed,
    ALBANIAN_LANG_ISO_CODE,
    ENGLISH_LANG_ISO_CODE
)
logging.info("Finished preprocessing.")
