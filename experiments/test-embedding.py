import torch
from clickhouse_driver import Client
from transformers import AutoModel, AutoTokenizer



client = Client('localhost')
print("Ensuring table exists")
client.execute("CREATE TABLE IF NOT EXISTS chatbot (chunk String, embedding Array(Float32)) ENGINE = MergeTree ORDER BY chunk")
print("Yes")


def get_embeddings(tokenizer, model, chunk):
    print("Generating embedding for " + chunk)
    inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    pooled_embedding = torch.mean(embeddings, dim=1)
    pooled_embedding = pooled_embedding.squeeze().numpy().tolist()
    print(pooled_embedding)
    return pooled_embedding


def get_embeddings_from_fresh_model(queries):
    print("Loading model")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    print("Loaded")
    return [get_embeddings(tokenizer, model, chunk) for chunk in queries]
    


print ("First, generating a couple of values")

chunk = "hello world"
embedding = get_embeddings_from_fresh_model([chunk])[0]
embedding = ','.join(map(str, embedding))
client.execute(f"INSERT INTO chatbot (chunk, embedding) VALUES ('{chunk}', [{embedding}])")

chunk = "hello there"
embedding = get_embeddings_from_fresh_model([chunk])[0]
embedding = ','.join(map(str, embedding))
client.execute(f"INSERT INTO chatbot (chunk, embedding) VALUES ('{chunk}', [{embedding}])")

print("Now querying")

chunk = "hello world"
embedding = get_embeddings_from_fresh_model([chunk])[0]
embedding = ','.join(map(str, embedding))
result = client.execute(f"SELECT chunk, L2Distance(embedding, [{embedding}]) AS score FROM chatbot")
print(result)
