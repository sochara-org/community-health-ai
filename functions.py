from clickhouse_driver import Client
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import torch
from nltk.tokenize import word_tokenize
import openai
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import sentencepiece
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM
import os
from dotenv import load_dotenv

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
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
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

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def rank_sections(client, question_embedding, question_tokens):
    try:
        sections = client.execute("SELECT b_chunks.chunk_text, b_chunks.embeddings, b_table.original_filename FROM b_chunks JOIN b_table ON b_chunks.summary_id = b_table.id")
        if not sections:
            print("No sections retrieved from the database.")
            return None
        ranked_sections = []
        question_token_set = set(question_tokens)

        for section in sections:
            if len(section) < 3:
                print("Invalid section format:", section)
                continue

            chunk_text = section[0]
            if isinstance(chunk_text, bytes):
                try:
                    chunk_text = chunk_text.decode('utf-8')
                except UnicodeDecodeError:
                    chunk_text = chunk_text.decode('utf-8', errors='replace')

            embeddings_list = section[1]
            original_filename = section[2]

            if not isinstance(embeddings_list, list):
                print("Embeddings are not in the expected list format.")
                continue

            if len(embeddings_list) != len(question_embedding):
                print("Dimension mismatch between query and chunk embeddings.")
                continue

            embeddings_tensor = torch.tensor(embeddings_list)
            question_embedding_tensor = torch.tensor(question_embedding)
            cosine_similarity = torch.nn.functional.cosine_similarity(question_embedding_tensor, embeddings_tensor, dim=0)

            chunk_tokens = set(word_tokenize(chunk_text.lower()))
            jaccard_similarity_score = jaccard_similarity(question_token_set, chunk_tokens)

            combined_similarity = (0.7 * cosine_similarity.item()) + (0.3 * jaccard_similarity_score)

            ranked_sections.append((chunk_text, combined_similarity, original_filename))

        ranked_sections.sort(key=lambda x: x[1], reverse=True)
        return ranked_sections

    except Exception as e:
        print("An error occurred while ranking sections:", e)
        return None
    
def cosine_similarity(client, question_embedding):
    try:
        question_embedding_str = ','.join(map(str, question_embedding))
        query = f"""
        SELECT chunk_text, original_filename,
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
        chunk_text, original_filename = sections[0][:2]
        return chunk_text, original_filename
    except Exception as e:
        print("An error occurred during vector search in chunks:", e)
        return None, None

def vector_search_cosine_distance(client, question_embedding):
    try:
        question_embedding_str = ','.join(map(str, question_embedding))
        query = f"""
        SELECT chunk_text, original_filename,
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
        chunk_text, original_filename = sections[0][:2]
        return chunk_text, original_filename
    except Exception as e:
        print("An error occurred during vector search in chunks:", e)
        return None, None

def ann_search(client, query_embedding):
    try:
        query_embedding_str = ','.join(map(str, query_embedding))
        query = f"""
        SELECT chunk_text, original_filename,
               1 - (dotProduct(embeddings, [{query_embedding_str}]) / 
                    (sqrt(dotProduct(embeddings, embeddings)) * sqrt(dotProduct([{query_embedding_str}], [{query_embedding_str}])))
               ) AS distance
        FROM b_chunks
        JOIN b_table ON b_chunks.summary_id = b_table.id
        ORDER BY distance ASC
        LIMIT 1
        """
        sections = client.execute(query)
        if not sections:
            print("No sections retrieved from the database.")
            return None, None
        chunk_text, original_filename = sections[0][:2]
        return chunk_text, original_filename
    except Exception as e:
        print("An error occurred during vector search in chunks:", e)
        return None, None

def euclidean_search(client, question_embedding):
    try:
        question_embedding_str = ','.join(map(str, question_embedding))
        query = f"""
        SELECT chunk_text, original_filename,
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
        chunk_text, original_filename = sections[0][:2]
        return chunk_text, original_filename
    except Exception as e:
        print("An error occurred during vector search in chunks:", e)
        return None, None


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

def process_query(query_text):
    tokenizer, model = initialize_tokenizer_and_model()
    client = initialize_clickhouse_connection()
    question_embedding = generate_embeddings(tokenizer, model, query_text)
    if question_embedding is not None:
        question_tokens = word_tokenize(query_text.lower())
        ranked_sections = rank_sections(client, question_embedding, question_tokens)
        if ranked_sections:
            nearest_chunk, similarity_score, pdf_filename = ranked_sections[0]
            structured_sentence = structure_sentence(query_text, nearest_chunk)
            return structured_sentence, pdf_filename
    return None, None

def process_query_clickhouse(query_text, search_method='euclidean_search'):
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
                structured_sentence = structure_chunk_text(query_text, chunk_text)
                return structured_sentence, pdf_filename
        else:
            print(f"Search method '{search_method}' is not valid.")
    return None, None

