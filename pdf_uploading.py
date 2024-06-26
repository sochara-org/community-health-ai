import os
from transformers import BertTokenizer, BertModel
import torch
import nltk
from nltk.tokenize import sent_tokenize
from clickhouse_driver import Client
from PyPDF2 import PdfReader
import json
import uuid
import pytesseract
from pdf2image import convert_from_path
import requests
from dotenv import load_dotenv


load_dotenv()

# Setup
nltk.download('punkt')

# ClickHouse connection parameters
host = os.getenv('CLICKHOUSE_HOST')
port = int(os.getenv('CLICKHOUSE_PORT'))
secure = True
password = os.getenv('CLICKHOUSE_PASSWORD')
database = os.getenv('CLICKHOUSE_DATABASE')

# Directory containing PDF files
pdf_directory = os.getenv('PDF_DIRECTORY')


# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Table schemas
table_schema = '''
CREATE TABLE IF NOT EXISTS abc_table (
    id UUID DEFAULT generateUUIDv4(),
    user_name String,
    original_filename String,
    summarized_text Nullable(String),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id;
'''

table_schema_chunks = '''
CREATE TABLE IF NOT EXISTS abc_chunks (
    id Int64,
    summary_id UUID,
    chunk_text String,
    embeddings Array(Float32),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
SETTINGS
    index_granularity = 8192,
    enable_mixed_granularity_parts = 1;
'''

index_creation_query = '''
CREATE INDEX IF NOT EXISTS hnsw_embeddings ON abc_chunks (embeddings) TYPE annoy GRANULARITY 1;
'''

# Create ClickHouse tables
def create_clickhouse_tables():
    try:
        client = Client(host=host, port=port, secure=secure, password=password, database=database)
        client.execute(table_schema)
        client.execute(table_schema_chunks)
        client.execute(index_creation_query)
        client.disconnect()
    except Exception as e:
        print("Error creating ClickHouse tables:", e)

def insert_pdf_summary(user_name, original_filename):
    try:
        client = Client(host=host, port=port, secure=secure, password=password, database=database)
        unique_id = uuid.uuid4()
        query = f"INSERT INTO abc_table (id, user_name, original_filename, summarized_text) VALUES ('{unique_id}', '{user_name}', '{original_filename}', NULL)"
        client.execute(query)
        client.disconnect()
        return unique_id
    except Exception as e:
        print("Error inserting PDF summary into ClickHouse:", e)
        return None

def extract_text_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        extracted_text = ''
        for image in images:
            extracted_text += pytesseract.image_to_string(image)
        return extracted_text
    except Exception as e:
        print(f"An error occurred while extracting text from PDF: {e}")
        return None

def insert_chunks(summary_id, pdf_text):
    try:
        client = Client(host=host, port=port, secure=secure, password=password, database=database)
        
        sentences = nltk.sent_tokenize(pdf_text)
        sentences_per_chunk = 4
        chunked_sentences = []

        # Group sentences into chunks
        current_chunk = []
        for sentence in sentences:
            current_chunk.append(sentence)
            if sentence.endswith('.'):
                if len(current_chunk) >= sentences_per_chunk:
                    chunked_sentences.append(current_chunk)
                    current_chunk = []

        # If there are remaining sentences, add them to the last chunk
        if current_chunk:
            chunked_sentences.append(current_chunk)

        for idx, chunk_sentences in enumerate(chunked_sentences, start=1):
            concatenated_chunk = " ".join(chunk_sentences)

            inputs = tokenizer(concatenated_chunk, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

            embedding_str = json.dumps(embedding)

            # Use a subquery to generate a unique ID for each new row
            query = f"INSERT INTO abc_chunks (id, summary_id, chunk_text, embeddings) SELECT COALESCE(MAX(id), 0) + 1, '{summary_id}', '{concatenated_chunk}', [{', '.join(map(str, embedding))}] FROM abc_chunks"

            client.execute(query)

            print(f"Processed chunk {idx} of {len(chunked_sentences)}")

        print("Chunks inserted into ClickHouse successfully.")
    except Exception as e:
        print("Error inserting document chunks into ClickHouse:", e)
    finally:
        client.disconnect()

def process_pdf_file(pdf_file_path, user_name):
    print(f"Processing PDF: {pdf_file_path}")
    try:
        # Check if the summary already exists in ClickHouse
        original_filename = os.path.basename(pdf_file_path)
        existing_summary_query = f"SELECT COUNT(*) FROM abc_table WHERE original_filename = '{original_filename}'"
        client = Client(host=host, port=port, secure=secure, password=password, database=database)
        existing_summary_count = client.execute(existing_summary_query)[0][0]
        client.disconnect()

        if existing_summary_count > 0:
            print("Summary already exists in ClickHouse. Skipping processing.")
            return

        # Extract text from the PDF
        pdf_text = ""
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() or ""

        # Split PDF text into smaller chunks based on a maximum number of characters
        max_chunk_characters = 1000
        chunks = []
        chunk_start = 0
        while chunk_start < len(pdf_text):
            chunk_end = chunk_start + max_chunk_characters
            while chunk_end < len(pdf_text) and not pdf_text[chunk_end].isspace():
                chunk_end += 1  # Extend chunk_end to next whitespace
            chunks.append(pdf_text[chunk_start:chunk_end])
            chunk_start = chunk_end

        # Insert the PDF summary into ClickHouse
        summary_id = insert_pdf_summary(user_name, original_filename)
        if summary_id:
            print("PDF summary inserted into ClickHouse successfully.")

            # Insert each chunk into ClickHouse
            for idx, chunk_text in enumerate(chunks, start=1):
                insert_chunks(summary_id, chunk_text)
                print(f"Processed chunk {idx} of {len(chunks)}")
        else:
            print("Failed to insert PDF summary into ClickHouse.")

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")

# Main function to process all PDF files in the directory
def main():
    try:
        # Create ClickHouse tables if they don't exist
        create_clickhouse_tables()

        # Process each PDF file in the directory
        for pdf_file in os.listdir(pdf_directory):
            if pdf_file.endswith(".pdf"):
                pdf_file_path = os.path.join(pdf_directory, pdf_file)
                process_pdf_file(pdf_file_path, "default")  # Pass 'default' as the user_name
    except Exception as e:
        print("An error occurred:", e)

# Execute the main function
if __name__ == "__main__":
    main()