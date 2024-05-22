from llama_index.core import GPTSimpleVectorIndex


# Load the vector store from disk
vector_store = SimpleVectorStore.load_from_disk('vector_store.json')

# Query the vector store
query = "What are the key points of the document?"
response = vector_store.query(query)

print(response)
