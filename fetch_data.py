"""
This script automates the download, extraction, and preprocessing of PTM (Post-Translational Modification) datasets
from the dbPTM database hosted by NYCU. The data is fetched as raw TSV files inside `.tgz` archives, then processed
into cleaned CSV files that include full UniProt protein sequences for each PTM entry.

Main Steps:
1. **Download Raw Data**:
   - Connects to the dbPTM download page.
   - Identifies and downloads `.tgz` archives for each PTM dataset (Linux/Mac versions).
   - Extracts the contents into the `ptms_raw/` directory.

2. **Preprocess and Format**:
   - Iterates through each raw TSV file.
   - Cleans whitespace from UniProt IDs.
   - Fetches full protein sequences for each UniProt accession using `fetch_accessions`.
   - Appends the sequence as a new column (`seq`) in the DataFrame.
   - Saves the result as a CSV file in the `ptms_clean/` directory.

Requirements:
- `bs4`, `requests`, `pandas`
- Custom modules: `neurosnap.log.logger` and `neurosnap.protein.fetch_accessions`

Directory Structure:
- `ptms_raw/` – Temporary directory for raw TSV files (created from downloaded archives).
- `ptms_clean/` – Final output directory for cleaned and enriched CSV files.

Usage:
Run this script directly. It will skip already-processed files to avoid redundant work.

Note:
- This script assumes consistent structure in the dbPTM HTML and TSV file formatting.
- All TSVs are expected to use tab separation and contain six columns.
"""

import os
import shutil
import tarfile

import bs4
import pandas as pd
import requests
from neurosnap.log import logger
from neurosnap.protein import fetch_accessions


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
os.makedirs(INPUTS_CLEAN_DIR, exist_ok=True)

## process each raw TSV file
# add full uniprot sequences to each dataframe
for fname in os.listdir(INPUTS_RAW_DIR):
  fpath_in = os.path.join(INPUTS_RAW_DIR, fname)
  fpath_out = os.path.join(INPUTS_CLEAN_DIR, fname + ".csv")
  if os.path.exists(fpath_out):  # skip finished files
    logger.info(f"Processing raw file {fname} (skipping)")
    continue

  df = pd.read_csv(fpath_in, sep="\t", names=["name", "uniprot", "ptm_location", "ptm_name", "unknown_stupid", "adjacent_seq"])
  df.uniprot = df.uniprot.str.strip()
  logger.info(f"Processing raw file {fname} ({len(df)} entries)")
  full_seqs = []
  accessions = fetch_accessions(df.uniprot)
  for _, row in df.iterrows():
    full_seqs.append(accessions[row.uniprot])

  df["seq"] = full_seqs
  df.to_csv(fpath_out, index=False)