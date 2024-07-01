import os
import re
import torch
import urllib.parse
import openai
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from clickhouse_driver import Client
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from stop_words import stop_words
import logging
import time
import functools
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

load_dotenv()
archive_base_url = os.getenv('ARCHIVE_BASE_URL')

if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

dotenv_path = '.env'

if not os.path.exists(dotenv_path):
    raise FileNotFoundError(f"The .env file at path '{dotenv_path}' does not exist.")

load_dotenv(dotenv_path)

openai.api_key = os.getenv('OPENAI_API_KEY')

def timeit(func):
    @functools.wraps(func)
    def wrapper_timeit(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        return result
    return wrapper_timeit

@timeit
def initialize_clickhouse_connection():
    return Client(host=os.getenv('CLICKHOUSE_HOST'),
                  port=int(os.getenv('CLICKHOUSE_PORT')),
                  secure=True,
                  password=os.getenv('CLICKHOUSE_PASSWORD'),
                  database=os.getenv('CLICKHOUSE_DATABASE'))

@timeit
@functools.lru_cache(maxsize=None)
def get_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    return tokenizer, model


@timeit
def generate_embeddings(tokenizer, model, query):
    try:
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        pooled_embedding = torch.mean(embeddings, dim=1)
        pooled_embedding = pooled_embedding.squeeze().numpy().tolist()
        return pooled_embedding
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

@timeit
def extract_important_words(query_text):
    words = re.findall(r'\b\w+\b', query_text.lower())
    important_words = [word for word in words if word not in stop_words]
    return important_words

@timeit
def get_surrounding_chunks_batch(client, chunk_ids, summary_ids, window_size=2):
    # Ensure chunk_ids and summary_ids are lists
    chunk_ids = [chunk_ids] if not isinstance(chunk_ids, (list, tuple)) else chunk_ids
    summary_ids = [summary_ids] if not isinstance(summary_ids, (list, tuple)) else summary_ids

    # Convert UUIDs and integers to strings
    chunk_ids = [str(id) for id in chunk_ids]
    summary_ids = [str(id) for id in summary_ids]

    # Ensure we have at least one item in each tuple for the IN clause
    chunk_ids_tuple = tuple(chunk_ids) if len(chunk_ids) > 1 else f"('{chunk_ids[0]}')"
    summary_ids_tuple = tuple(summary_ids) if len(summary_ids) > 1 else f"('{summary_ids[0]}')"

    query = f"""
    SELECT id, chunk_text, summary_id
    FROM abc_chunks
    WHERE summary_id IN {summary_ids_tuple}
      AND id >= (SELECT MIN(id) - {window_size} FROM abc_chunks WHERE id IN {chunk_ids_tuple})
      AND id <= (SELECT MAX(id) + {window_size} FROM abc_chunks WHERE id IN {chunk_ids_tuple})
    ORDER BY summary_id, id
    """
    results = client.execute(query)
    chunks = {}
    for id, chunk_text, summary_id in results:
        if summary_id not in chunks:
            chunks[summary_id] = []
        chunks[summary_id].append(chunk_text)
    return {str(summary_id): ' '.join(texts) for summary_id, texts in chunks.items()}

@timeit
@lru_cache(maxsize=1000)
def get_original_filename(client, summary_id):
    try:
        query = f"SELECT original_filename FROM abc_table WHERE id = '{summary_id}' LIMIT 1"
        result = client.execute(query)
        if result and result[0][0]:
            original_filename = result[0][0]
            original_filename = original_filename.split('None', 1)[-1].strip()
            filename_without_ext = os.path.splitext(original_filename)[0]
            parsed_url = urllib.parse.urlparse(filename_without_ext)
            filename = parsed_url.path.split('/')[-1]
            file_url = f"{archive_base_url}{filename}"
            return file_url
        else:
            return None
    except Exception as e:
        logger.error(f"Error fetching original filename: {e}")
        return None


@timeit
def ann_search(client, query_embedding, window_size=2, top_n=1):
    try:
        question_embedding_str = ','.join(map(str, query_embedding))
        query = f"""
        SELECT c.id, c.chunk_text, c.summary_id,
               (dotProduct(c.embeddings, [{question_embedding_str}]) /
               (sqrt(dotProduct(c.embeddings, c.embeddings)) * sqrt(dotProduct([{question_embedding_str}], [{question_embedding_str}])))
                ) AS cosine_similarity
        FROM abc_chunks AS c
        JOIN abc_table AS a ON c.summary_id = a.id
        ORDER BY cosine_similarity DESC
        LIMIT {top_n}
        """
        sections = client.execute(query)
        if not sections:
            logger.info("No sections retrieved from the database.")
            return None, None

        # Get details only for the top chunk
        top_id, top_chunk_text, top_summary_id, top_cosine_similarity = sections[0]
        full_context = get_surrounding_chunks_batch(client, [top_id], [top_summary_id], window_size)
        file_url = get_original_filename(client, top_summary_id)

        chunks = [(full_context.get(str(top_summary_id), ''), file_url)]
        pdf_filenames = [file_url] if file_url and file_url.endswith('.pdf') else []

        # Add other chunks without full context or file URL
        for section in sections[1:]:
            id, chunk_text, summary_id, cosine_similarity = section
            chunks.append((chunk_text, None))

        pdf_descriptions = [get_pdf_description(client, filename) for filename in pdf_filenames[:1]]
        return chunks, pdf_descriptions

    except Exception as e:
        logger.error(f"Error in ANN search: {e}")
        return None, None
    

@timeit
def query_clickhouse_word_with_multi_stage(client, important_words, query_embedding, top_n=1):
    query_embedding_str = ','.join(map(str, query_embedding))

    # Stage 1: Retrieve potentially relevant chunks based on keyword matching
    word_query = '%' + '%'.join(important_words) + '%'
    keyword_matching_query = f"""
    SELECT id, chunk_text, summary_id, embeddings
    FROM abc_chunks
    WHERE lower(chunk_text) LIKE lower('{word_query}')
    """
    matched_chunks = client.execute(keyword_matching_query)

    if matched_chunks:
        # Stage 2: Rank or re-rank the matched chunks using semantic similarity
        ranked_chunks_query = f"""
        SELECT c.id, c.chunk_text, c.summary_id,
               (dotProduct(c.embeddings, [{query_embedding_str}]) /
                (sqrt(dotProduct(c.embeddings, c.embeddings)) * sqrt(dotProduct([{query_embedding_str}], [{query_embedding_str}]))))
               AS cosine_similarity
        FROM (
        {keyword_matching_query}
        ) AS c
        JOIN abc_table AS a ON c.summary_id = a.id
        ORDER BY cosine_similarity DESC
        LIMIT {top_n}
        """
        ranked_chunks = client.execute(ranked_chunks_query)

        if ranked_chunks:
            # Get details only for the top chunk
            top_id, top_chunk_text, top_summary_id, top_cosine_similarity = ranked_chunks[0]
            full_context = get_surrounding_chunks_batch(client, [top_id], [top_summary_id])
            file_url = get_original_filename(client, top_summary_id)

            chunks = [(full_context.get(str(top_summary_id), ''), file_url)]
            pdf_filenames = [file_url] if file_url and file_url.endswith('.pdf') else []

            # Add other chunks without full context or file URL
            for chunk in ranked_chunks[1:]:
                id, chunk_text, summary_id, cosine_similarity = chunk
                chunks.append((chunk_text, None))

            # Retrieve descriptions for top PDF file
            pdf_descriptions = [get_pdf_description(client, filename) for filename in pdf_filenames[:1]]

            return chunks, pdf_descriptions

    # Fallback to ANN search if no relevant chunks found
    return ann_search(client, query_embedding, top_n=top_n)

@timeit
@lru_cache(maxsize=1000)
def get_pdf_description(client, filename):
    try:
        filename = os.path.basename(filename)
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        filename = filename.replace("None", "")

        query = f"SELECT id FROM abc_table WHERE original_filename = '{filename}'"
        result = client.execute(query)

        if result:
            summary_id = str(result[0][0])
            summary_id = summary_id.replace("'", "''")

            query_chunks = f"""
                SELECT chunk_text
                FROM abc_chunks
                WHERE summary_id = '{summary_id}'
                ORDER BY id ASC
                LIMIT 1
            """
            chunks_result = client.execute(query_chunks)
            
            if chunks_result:
               full_description = chunks_result[0][0]
            # Truncate the description to 300 characters and add ellipsis if needed
               description = (full_description[:247] + '...') if len(full_description) > 250 else full_description
               return description
            else:
                return "Description not found."
        else:
            return "File not found."

    except Exception as e:
        logger.error(f"Error querying ClickHouse: {e}")
        return "Error retrieving description."
    

@timeit
def get_random_filename(client):
    try:
        query = "SELECT original_filename FROM abc_table ORDER BY rand() LIMIT 1"
        result = client.execute(query)
        if result:
            original_filename = result[0][0]
            original_filename = original_filename.split('None', 1)[-1].strip()
            filename_without_ext = os.path.splitext(original_filename)[0]
            parsed_url = urllib.parse.urlparse(filename_without_ext)
            filename = parsed_url.path.split('/')[-1]
            file_url = f"{archive_base_url}{filename}"
            return file_url
        return None
    except Exception as e:
        print("An error occurred while fetching a random filename:", e)
        return None

@timeit
def deduplicate_results(client, closest_chunks, top_n=5):
    if not closest_chunks:
        return ["No content available"] * top_n, ["No file available"] * top_n

    full_contexts = []
    pdf_filenames = []
    seen_filenames = set()

    for chunk in closest_chunks:
        if len(full_contexts) >= top_n:
            break
        full_context, file_url = chunk
        if file_url not in seen_filenames:
            full_contexts.append(full_context)
            pdf_filenames.append(file_url)
            seen_filenames.add(file_url)

    while len(pdf_filenames) < top_n:
        random_filename = get_random_filename(client)
        if random_filename and random_filename not in seen_filenames:
            full_contexts.append("No additional unique content available")
            pdf_filenames.append(random_filename)
            seen_filenames.add(random_filename)

    return full_contexts[:top_n], pdf_filenames[:top_n]


@timeit
def structure_sentence_with_llama(query, chunk_text, llama_tokenizer, llama_model):
    try:
        input_prompt = f"Question: {query}\nAnswer: {chunk_text}"
        inputs = llama_tokenizer.encode(input_prompt, return_tensors='pt')
        outputs = llama_model.generate(inputs, max_length=100, temperature=0.01, top_p=1.0)
        completion_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completion_text.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

@timeit    
def get_structured_answer(query, chunk_text):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Your task is to find the most relevant information in the provided text to answer the user's query. Provide a concise and structured answer. But you shouldn't let them know that the text was provided to you. You should make it in such a way that the text was written by you. "},
            {"role": "user", "content": f"Query: {query}\n\nContext: {chunk_text}\n\nPlease provide a structured and concise answer to the query based on the given context."}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        structured_answer = response.choices[0].message.content
        return structured_answer.strip()
    except Exception as e:
        logger.error(f"Error in get_structured_answer: {str(e)}")
        return "I'm sorry, There seems to be no relevant answer for your question."

# Add debug prints and exception handling
@timeit
def process_query_clickhouse_pdf(query_text, top_n=5):
    try:
        tokenizer, model = get_tokenizer_and_model()
        client = initialize_clickhouse_connection()

        important_words = extract_important_words(query_text)
        if important_words:
            query_embedding = generate_embeddings(tokenizer, model, query_text)
            closest_chunks, _ = query_clickhouse_word_with_multi_stage(client, important_words, query_embedding, top_n=1)

            if closest_chunks:
                # Combine contexts for structured answer
                combined_context = " ".join([chunk for chunk, _ in closest_chunks])
                structured_answer = get_structured_answer(query_text, combined_context)
                
                # Use the structured answer as the only relevant chunk for further processing
                structured_chunks = [(structured_answer, closest_chunks[0][1])]
                
                full_contexts, pdf_filenames = deduplicate_results(client, structured_chunks, top_n=5)

                # Ensure uniqueness of pdf_filenames
                unique_filenames = list(dict.fromkeys(pdf_filenames))
                
                # If we don't have enough unique filenames, add random ones
                while len(unique_filenames) < top_n:
                    random_filename = get_random_filename(client)
                    if random_filename and random_filename not in unique_filenames:
                        unique_filenames.append(random_filename)

                pdf_descriptions = []
                for filename in unique_filenames:
                    if "No additional unique file" in filename:
                        pdf_descriptions.append("No additional unique description available")
                    else:
                        description = get_pdf_description(client, filename)
                        pdf_descriptions.append(description)

                # Ensure we have exactly top_n results
                full_contexts = full_contexts[:top_n]
                unique_filenames = unique_filenames[:top_n]
                pdf_descriptions = pdf_descriptions[:top_n]

                return full_contexts, unique_filenames, pdf_descriptions
            else:
                print("No closest_chunks found, returning placeholders")
                placeholder = "No content available"
                placeholder_file = "No file available"
                placeholder_desc = "No description available"
                return [placeholder]*top_n, [placeholder_file]*top_n, [placeholder_desc]*top_n

        return ["No content available"]*top_n, ["No file available"]*top_n, ["No description available"]*top_n

    except Exception as e:
        logger.error(f"Error in process_query_clickhouse_pdf: {str(e)}")
        return None, [], []  # Return default values or handle as needed
