"""
Read Ted Talk subtitles
"""
import glob
from pathlib import Path
import re
import pandas as pd

from collections import defaultdict

ALBANIAN_LANG_ISO_CODE = "sq"
ENGLISH_LANG_ISO_CODE = "en"

cwd = Path.cwd()
TED_TALKS_DIR = "ted-talks"
SQ_EN_SUBTITLES_DIR = "subs-en-sq"
JSON_EXTENSION = "json"

DATASET_DIR = cwd.joinpath(TED_TALKS_DIR, SQ_EN_SUBTITLES_DIR)


def files_in_dir(directory, extension):
    """
    For a give directory find all the files with the given extension
    :param directory:
    :param extension:
    :return:
    """
    path = Path.joinpath(directory, f"*.{extension}")
    file_paths = glob.glob(str(path))
    return file_paths


def get_videoids_by_language(files_list):
    """
    For all the subtitle files in the dataset, return the videoids grouped by language
    :param files_list: list of files in the dataset
    :return: dictionary of videos grouped by language
    """
    regex = re.compile(r'^VideoID-(\d+)-([a-z]{2})')
    file_stems_list = [Path(file).stem for file in files_list]
    stem_groups_list = [regex.search(file_stem) for file_stem in file_stems_list]
    stem_tuple_list = [(stem_groups.group(2), stem_groups.group(1)) for stem_groups in stem_groups_list]

    videos_by_language = defaultdict(list)

    for lang, videoid in stem_tuple_list:
        videos_by_language[lang].append(videoid)

    return videos_by_language


def get_videosids_intersection(video_ids_lang1, video_ids_lang2):
    """
    Get video ids for which subtitles are available in both languages
    :param video_ids_lang1:
    :param video_ids_lang2:
    :return:
    """
    intersection_set = set(video_ids_lang1) & set(video_ids_lang2)
    return list(intersection_set)


files_in_dir_list = files_in_dir(DATASET_DIR, JSON_EXTENSION)
videos_by_language_dict = get_videoids_by_language(files_in_dir_list)

sq_video_ids = videos_by_language_dict[ALBANIAN_LANG_ISO_CODE]
en_video_ids = videos_by_language_dict[ENGLISH_LANG_ISO_CODE]

videos_intersection = get_videosids_intersection(sq_video_ids, en_video_ids)
videos_intersection

sq = pd.read_json("ted-talks/subs-en-sq/VideoID-54-sq.json")
en = pd.read_json("ted-talks/subs-en-sq/VideoID-54-en.json")
