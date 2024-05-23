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

dotenv_path = 'api.env' 

if not os.path.exists(dotenv_path):
    raise FileNotFoundError(f"The .env file at path '{dotenv_path}' does not exist.")
# Load environment variables from .env file
load_dotenv(dotenv_path)

# Now you can access the API key as before
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
    # Update the path to match your local Llama model installation
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
    
def vector_search_in_chunks(client, question_embedding):
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
            return None
        return sections[0]  # Returning only the necessary parts (text and filename)
    except Exception as e:
        print("An error occurred during vector search in chunks:", e)
        return None
    
def vector_search_in_chunks_anb(client, query_text, question_embedding):
    try:
        question_embedding_str = ','.join(map(str, question_embedding))
        query = f"""
        SELECT chunk_text, embeddings, original_filename,
               (dotProduct(embeddings, [{question_embedding_str}]) / 
                (sqrt(dotProduct(embeddings, embeddings)) * sqrt(dotProduct([{question_embedding_str}], [{question_embedding_str}]))) ) AS cosine_similarity
        FROM b_chunks
        JOIN b_table ON b_chunks.summary_id = b_table.id
        ORDER BY cosine_similarity DESC
        LIMIT 10  -- Fetch more chunks to ensure we get neighbors
        """

        sections = client.execute(query)
        if not sections:
            print("No sections retrieved from the database.")
            return None

        # Find the top chunk and its neighbors
        top_chunk_index = 0
        closest_chunk = sections[top_chunk_index]

        # Retrieve the chunk above and below if they exist
        above_chunk = sections[top_chunk_index - 1] if top_chunk_index > 0 else None
        below_chunk = sections[top_chunk_index + 1] if top_chunk_index < len(sections) - 1 else None

        # Combine the chunks into one
        combined_chunk_text = ""
        if above_chunk:
            above_chunk_text, _, _, _ = above_chunk
            combined_chunk_text += above_chunk_text + " "
        combined_chunk_text += closest_chunk[0]  # Adding the closest chunk
        if below_chunk:
            below_chunk_text, _, _, _ = below_chunk
            combined_chunk_text += " " + below_chunk_text

        # Structure the combined chunk
        structured_combined_chunk = structure_chunk_text(query_text, combined_chunk_text)

        return structured_combined_chunk

    except Exception as e:
        print("An error occurred during vector search in chunks:", e)
        return None




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

def process_query_clickhouse(query_text):
    #tokenizer, model = initialize_tokenizer_and_model()
    tokenizer, model = initialize_tokenizer_and_model()
    client = initialize_clickhouse_connection()
    question_embedding = generate_embeddings(tokenizer, model, query_text)
    if question_embedding is not None:
        closest_chunk = vector_search_in_chunks(client, question_embedding)
        if closest_chunk:
            chunk_text, pdf_filename, _ = closest_chunk
            structured_sentence = structure_chunk_text(query_text,chunk_text)
            return structured_sentence, pdf_filename
    return None, None
