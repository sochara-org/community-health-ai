import sys
import json

from functions import *
from ai import *

def ingest_ia_document(id):
    html_path = f"files/{id}_djvu.txt"
    html_url = f"https://archive.org/download/{id}/{id}_djvu.txt"
    html_text = read_or_download_file(html_path, html_url)

    metadata_path = f"files/{id}.json"
    metadata_url = f"https://archive.org/metadata/{id}"
    metadata_string = read_or_download_file(metadata_path, metadata_url)
    metadata = json.loads(metadata_string)

    metadata = {k:v for k, v in metadata["metadata"].items() if k in ["creator", "publisher", "title"]}
    metadata["url"] = f"https://archive.org/details/{id}"

    text = extract_pre_text(html_text)
    paragraphs = get_paragraphs(text)
    chunks = convert_to_chunks(paragraphs)

    chunks = chunks[:100]
    print(len(chunks))

    result = create_document_and_add_chunks(metadata.get("title"), metadata, chunks)

    # result = chunk_state("corpora/community-health-ai-wtes3sbg1enp/documents/better-care-in-leprosy-r916mjnz6t5x")
    # print(result)

if __name__ == "__main__":
    ingest_ia_document(sys.argv[1])