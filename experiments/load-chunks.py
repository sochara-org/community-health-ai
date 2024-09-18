from common import get_tokenizer, get_model, get_client, generate_embeddizer, used_db, used_model

def ensure_table_exists(client):
    client.execute(f"CREATE TABLE IF NOT EXISTS {used_db} (chunk String, embedding Array(Float32)) ENGINE = MergeTree ORDER BY chunk")

client=get_client()
ensure_table_exists(client)
tokenizer = get_tokenizer(used_model)
model = get_model(used_model)

get_embedding = generate_embeddizer(tokenizer, model)

pdf = """India is my country and all Indians are my brothers and sisters.I love my country and I am proud of its rich and varied heritage.I shall always strive to be worthy of it.I shall give my parents, teachers and all elders respect and treat everyone with courtesy.To my country and my people, I pledge my devotion. In their well-being and prosperity alone lies my happiness."""
# pdf = "Hello world.Hello there"
chunks = pdf.split('.')
chunks = [chunk for chunk in chunks if chunk != '']

embeddings = [get_embedding(chunk) for chunk in chunks]
embeddings = [','.join(map(str, embedding)) for embedding in embeddings]
query = f"INSERT INTO {used_db} (chunk, embedding) VALUES"
for chunk, embedding in zip(chunks, embeddings):
    query += f"('{chunk}', [{embedding}])"
client.execute(query)
