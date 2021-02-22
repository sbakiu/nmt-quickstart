import pandas as pd
import glob
import os
import csv
from pathlib import Path

from mosestokenizer import MosesTokenizer, MosesDetokenizer


def write_detok_test_set(split_directory):
    """Read tokenized dataset from 'split_directory', create detokenized dataset
      and store them in the same directory.
    """
    file_paths = glob.glob(os.path.join(split_directory, '*.test.csv'))
    test_filepath = file_paths[0]
    print(f'Read csv file from {test_filepath}')
    test_df = pd.read_csv(test_filepath, encoding='utf-8')

    en_tokenizer = MosesTokenizer()
    en_detokenizer = MosesDetokenizer()

    test_df['en'] = test_df['en'].apply(lambda x: en_detokenizer(en_tokenizer(x)))
    test_df['sq'] = test_df['sq'].apply(lambda x: en_detokenizer(en_tokenizer(x)))
    # test_df['th'] = test_df['th'].apply(lambda x: ' '.join(th_word_space_tokenize(x))).apply(th_detokenize)

    test_df[['en']].to_csv(os.path.join(split_directory, 'test.detok.en'), encoding='utf-8', sep="\t", index=False, header=False, escapechar="", quotechar="", quoting=csv.QUOTE_NONE)
    test_df[['sq']].to_csv(os.path.join(split_directory, 'test.detok.sq'), encoding='utf-8', sep="\t", index=False, header=False, escapechar="", quotechar="", quoting=csv.QUOTE_NONE)

    print('Done writing test set into text files.')


def remove_bpe(input_file_path, output_file_path):
    """Remove PBE boundary marker of the given file and write the output to the given path """

    with open(input_file_path, 'r') as reader, open(output_file_path, encoding='utf-8', mode='w') as writer:
        out = []
        for line in reader.readlines():

            out.append(''.join(map(lambda x: x.replace(
                '▁', ' '), line.split(' '))).lstrip(' '))

        writer.writelines(out)


cwd = Path.cwd()
input_path = cwd.joinpath("hypo.sq-en.sys")
output_path = str(input_path) + ".removed_bpe"
remove_bpe(input_path, output_path)

cwd = Path.cwd()
input_path = cwd.joinpath("ted-talks", "split", "combined-subs-sq-en")
write_detok_test_set(str(input_path))
