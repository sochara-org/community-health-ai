import os
import requests
from dotenv import load_dotenv

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file
dotenv_path = os.path.join(script_dir, '.env')
# Load the .env file
load_dotenv(dotenv_path)

# Then use os.getenv as before
metadata_url = os.getenv('METADATA_URL')
output_directory = os.getenv('OUTPUT_DIRECTORY')
archive_base_url = os.getenv('ARCHIVE_BASE_URL')

# Function to download PDF files
def download_pdfs_from_metadata(metadata_url, output_dir):
    # Fetch metadata from the API
    response = requests.get(metadata_url)
    if response.status_code != 200:
        print("Failed to fetch metadata.")
        return

    # Parse JSON response
    metadata = response.json()

    # Extract PDF file names
    pdf_identifiers = []
    for item in metadata['response']['docs']:
        if 'identifier' in item:
            pdf_identifiers.append(item['identifier'])

    # Construct PDF URLs and download each PDF
    for pdf_id in pdf_identifiers:
        pdf_url = f'{archive_base_url}{pdf_id}/{pdf_id}.pdf'
        pdf_filename = os.path.join(output_dir, f"{pdf_id}.pdf")
        # Download PDF
        download_pdf(pdf_url, pdf_filename)

# Function to download a PDF file
def download_pdf(pdf_url, pdf_filename):
    with open(pdf_filename, 'wb') as f:
        pdf_content = requests.get(pdf_url)
        f.write(pdf_content.content)
        print(f"Downloaded: {pdf_filename}")

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Download PDFs
download_pdfs_from_metadata(metadata_url, output_directory)
