This project implements a query processing and vector search application using various machine learning models and a ClickHouse database. The application processes queries, generates embeddings, and ranks or retrieves relevant sections from a database.

## Features

- Connection to a ClickHouse database
- Initialization of various tokenizers and models
- Generation of embeddings for input queries
- Ranking of database sections based on cosine similarity and Jaccard similarity
- Retrieval of database sections using different search methods (cosine distance, Euclidean distance, etc.)
- Structuring of text using language models (OpenAI GPT-3.5 and LLaMA-2)

## Installation

### Prerequisites

- Python 3.7 or higher
- ClickHouse server
- Necessary environment variables stored in a `.env` file

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/aparna23na/Proj.git
   cd your-repo-name
   
2. Create a virtual environment and activate it:
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

3. Set up the .env file with the necessary environment variables:
   OPENAI_API_KEY=your_openai_api_key
   CLICKHOUSE_HOST=your_clickhouse_host
   CLICKHOUSE_PORT=your_clickhouse_port
   CLICKHOUSE_USER=your_clickhouse_user
   CLICKHOUSE_PASSWORD=your_clickhouse_password
   CLICKHOUSE_DATABASE=your_clickhouse_database


By default, the application will be accessible at http://127.0.0.1:5000/.

Flask Application
Home Page: Enter a query to receive a structured response based on the most relevant section from the database.
About Page: Information about the project.

Key Functions:
functions.py
initialize_clickhouse_connection: Establishes a connection to the ClickHouse database.

initialize_tokenizer_and_model: Initializes a tokenizer and model (e.g., GPT-2).

initialize_llama_model: Initializes the LLaMA-2 model.

generate_embeddings: Generates embeddings for a given query.

rank_sections: Ranks sections from the database based on cosine and Jaccard similarity.

cosine_similarity: Retrieves the most similar section based on cosine similarity
.
vector_search_cosine_distance: Retrieves the most similar section based on cosine distance.

ann_search: Retrieves the most similar section using approximate nearest neighbors search.

euclidean_search: Retrieves the most similar section based on Euclidean distance.

structure_sentence_with_llama: Structures text using the LLaMA-2 model.

structure_sentence: Structures text using OpenAI's GPT-3.5 model.

structure_chunk_text: Structures text without adding any changes.

process_query: Processes a query and returns a structured response.

process_query_clickhouse: Processes a query using different search methods and returns a structured response.


search_query.py:
index: Renders the home page and processes user queries.
about: Renders the about page.

   



