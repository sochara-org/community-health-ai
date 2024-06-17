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

load_dotenv()
archive_base_url = os.getenv('ARCHIVE_BASE_URL')

print("Current Working Directory:", os.getcwd())

if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

dotenv_path = 'a.env'

if not os.path.exists(dotenv_path):
    raise FileNotFoundError(f"The .env file at path '{dotenv_path}' does not exist.")

load_dotenv(dotenv_path)

openai.api_key = os.getenv('OPENAI_API_KEY')

def initialize_clickhouse_connection():
    return Client(host=os.getenv('CLICKHOUSE_HOST'),
                  port=int(os.getenv('CLICKHOUSE_PORT')),
                  secure=True,
                  password=os.getenv('CLICKHOUSE_PASSWORD'),
                  database=os.getenv('CLICKHOUSE_DATABASE'))

def initialize_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    return tokenizer, model

def initialize_llama_model():
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    return llama_tokenizer, llama_model

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
        print("An error occurred while converting query to embeddings:", e)
        return None
    
def extract_important_words(query_text):
    words = re.findall(r'\b\w+\b', query_text.lower())
    important_words = [word for word in words if word not in stop_words]
    return important_words


def get_surrounding_chunks(client, id, summary_id, window_size=2):
    surrounding_chunks_query = f"""
    SELECT chunk_text
    FROM abc_chunks
    WHERE summary_id = '{summary_id}' AND
          id >= {id - window_size} AND
          id <= {id + window_size}
    ORDER BY id
    """
    surrounding_chunks = client.execute(surrounding_chunks_query)
    full_context = ' '.join([chunk[0] for chunk in surrounding_chunks])
    return full_context

def get_original_filename(client, summary_id):
    # Execute the query to get the original filename
    query = f"SELECT original_filename FROM abc_table WHERE id = '{summary_id}'"
    result = client.execute(query)

    if result:
        original_filename = result[0][0]

        # Process the filename to remove 'None' prefix and strip whitespace
        original_filename = original_filename.split('None', 1)[-1].strip()

        # Extract the filename without the extension
        filename_without_ext = os.path.splitext(original_filename)[0]

        # Parse the URL to extract the filename
        parsed_url = urllib.parse.urlparse(filename_without_ext)
        filename = parsed_url.path.split('/')[-1]

        # Construct the full file URL
        file_url = f"{archive_base_url}{filename}"

        return file_url

    return None


def cosine_similarity(client, question_embedding):
    try:
        question_embedding_str = ','.join(map(str, question_embedding))
        query = f"""
        SELECT chunk_text, summary_id,
               (dotProduct(embeddings, [{question_embedding_str}]) / 
                (sqrt(dotProduct(embeddings, embeddings)) * sqrt(dotProduct([{question_embedding_str}], [{question_embedding_str}]))) ) AS cosine_similarity
        FROM b_chunks
        JOIN b_table ON b_chunks.summary_id = b_table.id
        ORDER BY cosine_similarity DESC
        LIMIT 1
        """
        sections = client.execute(query)
        if not sections:
            print("No sections retrieved from the database.")
            return None, None
        chunk_text, summary_id = sections[0][:2]
        original_filename = get_original_filename(client, summary_id)
        return chunk_text, original_filename
    except Exception as e:
        print("An error occurred during vector search in chunks:", e)
        return None, None

def vector_search_cosine_distance(client, question_embedding):
    try:
        question_embedding_str = ','.join(map(str, question_embedding))
        query = f"""
        SELECT chunk_text, summary_id,
               1 - (dotProduct(embeddings, [{question_embedding_str}]) / 
                    (sqrt(dotProduct(embeddings, embeddings)) * sqrt(dotProduct([{question_embedding_str}], [{question_embedding_str}])))
               ) AS cosine_distance
        FROM b_chunks
        JOIN b_table ON b_chunks.summary_id = b_table.id
        ORDER BY cosine_distance ASC
        LIMIT 1
        """
        sections = client.execute(query)
        if not sections:
            print("No sections retrieved from the database.")
            return None, None
        chunk_text, summary_id = sections[0][:2]
        original_filename = get_original_filename(client, summary_id)
        return chunk_text, original_filename
    except Exception as e:
        print("An error occurred during vector search in chunks:", e)
        return None, None

def ann_search(client, query_embedding, window_size=2):
    try:
        question_embedding_str = ','.join(map(str, query_embedding))
        query = f"""
        SELECT id, chunk_text, summary_id, embeddings,
               (dotProduct(embeddings, [{question_embedding_str}]) /
               (sqrt(dotProduct(embeddings, embeddings)) * sqrt(dotProduct([{question_embedding_str}], [{question_embedding_str}])))
               ) AS cosine_similarity
        FROM abc_chunks
        JOIN abc_table ON abc_chunks.summary_id = abc_table.id
        ORDER BY cosine_similarity DESC
        LIMIT 1
        """
        sections = client.execute(query)
        if not sections:
            print("No sections retrieved from the database.")
            return None, None
        
        id, chunk_text, summary_id, embeddings, cosine_similarity = sections[0]

        # Retrieve surrounding chunks based on the nearest chunk's id
        full_context = get_surrounding_chunks(client, id, summary_id, window_size)
        original_filename = get_original_filename(client, summary_id)
        return full_context, original_filename
    except Exception as e:
        print("An error occurred during vector search in chunks:", e)
        return None, None



def euclidean_search(client, question_embedding):
    try:
        question_embedding_str = ','.join(map(str, question_embedding))
        query = f"""
        SELECT chunk_text, summary_id,
               LpDistance(embeddings, [{question_embedding_str}], 2) AS euclidean_distance
        FROM b_chunks
        JOIN b_table ON b_chunks.summary_id = b_table.id
        ORDER BY euclidean_distance ASC
        LIMIT 1
        """
        sections = client.execute(query)
        if not sections:
            print("No sections retrieved from the database.")
            return None, None
        chunk_text, summary_id = sections[0][:2]
        original_filename = get_original_filename(client, summary_id)
        return chunk_text, original_filename
    except Exception as e:
        print("An error occurred during vector search in chunks:", e)
        return None, None


def query_clickhouse_word_with_multi_stage(client, important_words, query_embedding):

    tokenizer, model = initialize_tokenizer_and_model()
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
        SELECT c.id, c.chunk_text, c.summary_id, c.embeddings,
               (dotProduct(c.embeddings, [{query_embedding_str}]) /
                (sqrt(dotProduct(c.embeddings, c.embeddings)) * sqrt(dotProduct([{query_embedding_str}], [{query_embedding_str}]))))
               AS cosine_similarity
        FROM (
        {keyword_matching_query}
        ) AS c
        ORDER BY cosine_similarity DESC
        LIMIT 1
        """
        ranked_chunks = client.execute(ranked_chunks_query)

        if ranked_chunks:
            id, chunk_text, summary_id, embeddings, cosine_similarity = ranked_chunks[0]
            full_context = get_surrounding_chunks(client, id, summary_id)
            original_filename = get_original_filename(client, summary_id)
            return full_context, original_filename

 
    return ann_search(client, query_embedding)

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

def structure_sentence(query, chunk_text):
    try:
        messages = [
            {"role": "system", "content": f"Question: {query}"},
            {"role": "system", "content": f"Answer: {chunk_text}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
            temperature=0.01,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=[]
        )
        completion_text = response.choices[0].message.content
        return completion_text.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def structure_chunk_text(query, chunk_text):
    try:
        messages = [
            {"role": "system", "content": "Just format the text without adding any changes and removing any text"},
            {"role": "user", "content": f"Please structure the following text:\n\n{chunk_text}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,  # Increase max_tokens if you expect longer outputs
            temperature=0.01,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None
        )
        completion_text = response.choices[0].message.content
        return completion_text.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_query_clickhouse(query_text, search_method='ann_search'):
    tokenizer, model = initialize_tokenizer_and_model()
    client = initialize_clickhouse_connection()
    question_embedding = generate_embeddings(tokenizer, model, query_text)

    if question_embedding is not None:

        search_methods = {
            'cosine_similarity': cosine_similarity,
            'vector_search_cosine_distance': vector_search_cosine_distance,
            'ann_search': ann_search,
            'euclidean_search': euclidean_search
        }

        if search_method in search_methods:
            search_function = search_methods[search_method]
            chunk_text, pdf_filename = search_function(client, question_embedding)
            if chunk_text and pdf_filename:
                return chunk_text, pdf_filename
        else:
            print(f"Search method '{search_method}' is not valid.")
    return None, None

def process_query_clickhouse_word(query_text):
    tokenizer, model = initialize_tokenizer_and_model()
    client = initialize_clickhouse_connection()
    important_words = extract_important_words(query_text)

    if important_words:
        query_embedding = generate_embeddings(tokenizer, model, query_text)
        closest_chunks = query_clickhouse_word_with_multi_stage(client, important_words, query_embedding)
        if closest_chunks:
            full_context, original_filename = closest_chunks
            return full_context, original_filename

    print("No relevant chunk found. Performing ann_search as fallback.")
    return ann_search(client, query_embedding)
