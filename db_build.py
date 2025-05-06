"""
# API NOTES:
curl -X GET "https://rest.uniprot.org/uniprotkb/stream?format=json&query=((cc_ptm:*))"


curl -X GET "https://rest.uniprot.org/uniprotkb/search?format=json&fields=accession,features&query=((cc_ptm:*))"
curl -X GET "https://rest.uniprot.org/uniprotkb/search?format=json&fields=accession,sequence&query=ft_mod_res:*"


curl -X GET "https://rest.uniprot.org/uniprotkb/search?query=(cc_ptm:* OR ft_mod_res:*)&format=json"
"""

import json
import re

import requests


def get_next_link(link_header):
  if not link_header:
    return None
  match = re.search(r'<(.+?)>; rel="next"', link_header)
  return match.group(1) if match else None


# NOTE: Might be faster to switch to rest.uniprot.org/uniprotkb/stream endpoint
url = "https://rest.uniprot.org/uniprotkb/search"
params = {
  "query": "(cc_ptm:* OR ft_mod_res:* OR ft_lipid:*)",
  "format": "json",
  "size": 500,
}

i = 1
with open("ptms.jsonl", "w") as outfile:
  while url:
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    for entry in data.get("results", []):
      acc = entry["primaryAccession"]
      seq = entry.get("sequence", {}).get("value", "")
      mods = []

      for feature in entry.get("features", []):
        if feature.get("type", "").lower() == "modified residue":
          pos = feature.get("location", {}).get("start", {}).get("value")
          desc = feature.get("description", "")
          mods.append([desc, pos, pos])

      # Write one JSON object per line
      json_line = json.dumps({"acc": acc, "seq": seq, "mods": mods})
      outfile.write(json_line + "\n")

      print(f"{i}: {acc} | {seq[:10]} | {len(mods)} mods")
      i += 1

    # Prepare for the next page
    url = get_next_link(resp.headers.get("Link"))
    params = None  # only on the first request
