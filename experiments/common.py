import torch

from clickhouse_driver import Client
from transformers import AutoModel, AutoTokenizer

used_db = 'chatbot_bert'
used_model = 'bert-base-uncased'
    
def get_tokenizer(config='gpt2'):
    tokenizer = AutoTokenizer.from_pretrained(config)
    # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    return tokenizer

def get_model(config='gpt2'):
    return AutoModel.from_pretrained(config)

def get_client():
    return Client('localhost')

def get_embedding(tokenizer, model, chunk):
    inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    pooled_embedding = torch.mean(embeddings, dim=1)
    pooled_embedding = pooled_embedding.squeeze().numpy().tolist()
    return pooled_embedding

def generate_embeddizer(tokenizer, model):
    def get_embed(chunk):
        return get_embedding(tokenizer, model, chunk)
    return get_embed