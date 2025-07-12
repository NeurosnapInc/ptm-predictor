from Bio import Entrez, GenBank
import csv
import os
import time
import http.client
from urllib.error import HTTPError
import sys

os.makedirs('proteins', exist_ok=True)

Entrez.email = "leon.koh@rheasfactory.com"
Entrez.api_key = "bc98c30e5af58102dc2217a47ededa390608"

search_handle = Entrez.esearch(
    db="protein",
    term="txid10239[Organism:exp] NOT txid2697049[Organism:exp]",
    retmax=0,
    usehistory="y"
)
search_results = Entrez.read(search_handle)
search_handle.close()

total_proteins = int(search_results["Count"])
webenv = search_results["WebEnv"]
query_key = search_results["QueryKey"]
print(total_proteins)

batch_size = 1000
proteins_per_csv = 100000

headers = [
    "locus",
    "size",
    "residue_type",
    "data_file_division",
    "accession",
    "version",
    "db_source",
    "source",
    "organism",
    "taxonomy",
    "features_protein",
    "features_source",
    "features_cds",
    "origin",
    "sequence",
    "definition",
    "molecule_type",
    "topology"
]

csv_file_num = 121
csv_row_count = 0
csv_writer = None
csv_file = None

for start in range(12000000, total_proteins, batch_size):
    if csv_row_count == 0:
        if csv_file:
            csv_file.close()
        csv_filename = os.path.join('proteins', f"protein_{csv_file_num * proteins_per_csv}.csv")
        csv_file = open(csv_filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)
        print(f"Created {csv_filename}")

    current_batch = min(batch_size, total_proteins - start)

    max_retries = 2
    retry_count = 0
    while True:
        try:
            handle = Entrez.efetch(
                db="protein",
                query_key=query_key,
                WebEnv=webenv,
                retstart=start,
                retmax=current_batch,
                rettype="gb",
                retmode="text"
            )

            for record in GenBank.parse(handle):
                features = getattr(record, 'features', [])
                features_protein = [feature for feature in features if getattr(feature, 'key', '') == 'Protein']
                features_protein_str = "\n\n".join(str(feature) for feature in features_protein)
                features_source = [feature for feature in features if getattr(feature, 'key', '') == 'source']
                features_source_str = "\n\n".join(str(feature) for feature in features_source)
                features_cds = [feature for feature in features if getattr(feature, 'key', '') == 'CDS']
                features_cds_str = "\n\n".join(str(feature) for feature in features_cds)
                row = [
                    getattr(record, 'locus', ''),
                    getattr(record, 'size', ''),
                    getattr(record, 'residue_type', ''),
                    getattr(record, 'data_file_division', ''),
                    getattr(record, 'accession', ''),
                    getattr(record, 'version', ''),
                    getattr(record, 'db_source', ''),
                    getattr(record, 'source', ''),
                    getattr(record, 'organism', ''),
                    getattr(record, 'taxonomy', ''),
                    features_protein_str,
                    features_source_str,
                    features_cds_str,
                    getattr(record, 'origin', ''),
                    getattr(record, 'sequence', ''),
                    getattr(record, 'definition', ''),
                    getattr(record, 'molecule_type', ''),
                    getattr(record, 'topology', '')
                ]
                continue
                csv_writer.writerow(row)
                csv_row_count += 1

                if csv_row_count >= proteins_per_csv:
                    csv_file_num += 1
                    csv_row_count = 0

            handle.close()
            break
        except Exception as e:
            if isinstance(e, ValueError) and "binary mode not text mode" in str(e):
                print(f"Detected binary handle at position {start}, attempting to decode...")
                handle.seek(0)
                binary_content = handle.read()
                decoded_content = binary_content.decode('utf-8')
                import io
                text_handle = io.StringIO(decoded_content)
                import csv
                with open('error_test.csv', 'w', newline='') as error_test_file:
                    error_csv_writer = csv.writer(error_test_file)
                    header = [
                        'locus', 'size', 'residue_type', 'data_file_division', 'accession', 
                        'version', 'db_source', 'source', 'organism', 'taxonomy', 
                        'features_protein', 'features_source', 'features_cds', 'origin', 
                        'sequence', 'definition', 'molecule_type', 'topology'
                    ]
                    error_csv_writer.writerow(header)
                    for record in GenBank.parse(text_handle):
                        features = getattr(record, 'features', [])
                        features_protein = [feature for feature in features if getattr(feature, 'key', '') == 'Protein']
                        features_protein_str = "\n\n".join(str(feature) for feature in features_protein)
                        features_source = [feature for feature in features if getattr(feature, 'key', '') == 'source']
                        features_source_str = "\n\n".join(str(feature) for feature in features_source)
                        features_cds = [feature for feature in features if getattr(feature, 'key', '') == 'CDS']
                        features_cds_str = "\n\n".join(str(feature) for feature in features_cds)
                        row = [
                            getattr(record, 'locus', ''),
                            getattr(record, 'size', ''),
                            getattr(record, 'residue_type', ''),
                            getattr(record, 'data_file_division', ''),
                            getattr(record, 'accession', ''),
                            getattr(record, 'version', ''),
                            getattr(record, 'db_source', ''),
                            getattr(record, 'source', ''),
                            getattr(record, 'organism', ''),
                            getattr(record, 'taxonomy', ''),
                            features_protein_str,
                            features_source_str,
                            features_cds_str,
                            getattr(record, 'origin', ''),
                            getattr(record, 'sequence', ''),
                            getattr(record, 'definition', ''),
                            getattr(record, 'molecule_type', ''),
                            getattr(record, 'topology', '')
                        ]
                        error_csv_writer.writerow(row)
                    sys.exit(0)
                    break
            retry_count += 1
            if retry_count <= max_retries:
                print(f"Error fetching records at position {start}: {str(e)}")
                print("Waiting 10 minutes before retrying...")
                time.sleep(600)
            else:
                print(f"Failed after {max_retries} retries at position {start}. Moving to next batch.")
                csv_row_count += current_batch
                if csv_row_count >= proteins_per_csv:
                    csv_file_num += 1
                    csv_row_count = 0
                break
    print(f"Processed {min(start + batch_size, total_proteins)}/{total_proteins}")

if csv_file:
    csv_file.close()