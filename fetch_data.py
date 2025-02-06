import os
import shutil
import tarfile
from time import sleep

import bs4
import pandas as pd
import requests
from neurosnap.log import logger
from neurosnap.protein import fetch_uniprot
from tqdm import tqdm

### Constants
INPUTS_RAW_DIR = "ptms_raw"
INPUTS_CLEAN_DIR = "ptms_clean"

### Functions
def download_raw():
  ## download raw data
  # prepare directory
  shutil.rmtree(INPUTS_RAW_DIR, ignore_errors=True)
  os.makedirs(INPUTS_RAW_DIR)

  # fetch and parse download page
  r = requests.get("https://biomics.lab.nycu.edu.tw/dbPTM/download.php")
  r.raise_for_status()
  soup = bs4.BeautifulSoup(r.text)

  # download each ptm file
  for el in soup.select("#site .btn.btn-primary"):
    if el.text == "MAC / Linux":
      name = el["href"].split("/")[-1]
      logger.info(f"Fetching {name}")
      r = requests.get("https://biomics.lab.nycu.edu.tw/dbPTM/" + el["href"])
      r.raise_for_status()
      tgz_path = os.path.join(INPUTS_RAW_DIR, name)

      # Save the .tgz file
      with open(tgz_path, "wb") as f:
        f.write(r.content)

      # Extract the .tgz file
      with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(INPUTS_RAW_DIR)

      os.remove(tgz_path)

### Core Program
## prepare directories
# create raw files directory if it hasn't been fully downloaded yet
if not os.path.exists(INPUTS_CLEAN_DIR):
  download_raw()

# create clean directory
shutil.rmtree(INPUTS_CLEAN_DIR, ignore_errors=True)
os.makedirs(INPUTS_CLEAN_DIR)

## process each raw TSV file
# add full uniprot sequences to each dataframe
for fname in os.listdir(INPUTS_RAW_DIR):
  logger.info(f"Processing raw file {fname}")
  fpath_in = os.path.join(INPUTS_RAW_DIR, fname)
  fpath_out = os.path.join(INPUTS_CLEAN_DIR, fname + ".csv")
  df = pd.read_csv(fpath_in, sep="\t", names=["name", "uniprot", "ptm_location", "ptm_name", "unknown_stupid", "adjacent_seq"])
  full_seqs = []
  for _, row in tqdm(df.iterrows()):
    for _ in range(50):
      try:
        full_seqs.append(fetch_uniprot(row.uniprot))
        break
      except:
        sleep(5)
  df["seq"] = full_seqs
  df.to_csv(fpath_out, index=False)