import opustools
from pathlib import Path

cwd = Path.cwd()
# opus_get = opustools.OpusGet(
#     source='sq',
#     target='en',
#     directory='OpenSubtitles',
#     list_resources=True,
#     download_dir=str(cwd)
# )

opus_reader = opustools.OpusRead(
    source='sq',
    target='en',
    directory='OpenSubtitles',
    root_directory=str(cwd),
    write=["result.tmx"],
    write_mode="tmx",
    leave_non_alignments_out=False,
)
opus_reader.printPairs()
