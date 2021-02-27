import logging
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

with open("en-sq.tmx") as fp:
    content = fp.readlines()
    content = "".join(content)
    soup = BeautifulSoup(content, "lxml-xml")

body = soup.body
ens = body.findAll("tuv", {"xml:lang": "en"})
sqs = body.findAll("tuv", {"xml:lang": "sq"})

parsed = [(str(en.seg.string), str(sq.seg.string)) for en, sq in zip(ens, sqs)]
# create DataFrame using data
df = pd.DataFrame(parsed, columns=["en", "sq"])
df.to_csv("en-sq_opensubtitles.tsv", index=False, sep="\t")
df