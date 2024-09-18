import sys
from common import get_tokenizer, get_model, get_client, generate_embeddizer, used_db, used_model

client=get_client()
tokenizer = get_tokenizer(used_model)
model = get_model(used_model)

get_embedding = generate_embeddizer(tokenizer, model)


chunk = sys.argv[1] if len(sys.argv) > 1 else "India is my country and all Indians are my brothers and sisters"
embedding = get_embedding(chunk)
embedding = ','.join(map(str, embedding))

print("L2Distance")
result = client.execute(f"SELECT chunk, L2Distance(embedding, [{embedding}]) AS score FROM {used_db} ORDER BY score ASC")
for row in result:
    print(row)

print("\n 1 - cosineDistance")
result = client.execute(f"SELECT chunk, 1 - cosineDistance(embedding, [{embedding}]) AS score FROM {used_db} ORDER BY score DESC")
for row in result:
    print(row)
