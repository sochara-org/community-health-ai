This project implements a query processing and vector search application using various machine learning models and a ClickHouse database. The application processes queries, generates embeddings, and ranks or retrieves relevant sections from a database.

## Features

- Connection to a ClickHouse database
- Initialization of various tokenizers and models
- Generation of embeddings for input queries
- Ranking of database sections based on cosine similarity
- Retrieval of database sections using different search methods (cosine distance, Euclidean distance, etc.)
- Structuring of text using language model OpenAI GPT-3.5 

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
initialize_clickhouse_connection()
Initializes a connection to the ClickHouse database using the credentials and configurations stored in environment variables.

initialize_tokenizer_and_model()
Loads a pre-trained BERT tokenizer and model from the Hugging Face library. If the tokenizer lacks a padding token, it assigns a default one.

initialize_llama_model()
Loads a pre-trained LLaMA tokenizer and model from the Hugging Face library for causal language modeling tasks.

generate_embeddings(tokenizer, model, query)
Generates embeddings for a given query text using the provided tokenizer and model. Returns the pooled embedding vector.

extract_important_words(query_text)
Extracts important words from a given query text, excluding common stop words. Returns a list of significant words.

get_surrounding_chunks(client, id, summary_id, window_size=2)
Fetches chunks of text surrounding a specific chunk from the database. The window_size parameter determines how many chunks before and after should be included.

get_original_filename(client, summary_id)
Retrieves the original filename associated with a given summary ID from the database, processes it, and constructs the full file URL.

cosine_similarity(client, question_embedding)
Performs a cosine similarity search using the provided question embedding. Returns the most similar chunk of text and the associated original filename.

vector_search_cosine_distance(client, question_embedding)
Performs a cosine distance search using the provided question embedding. Returns the most similar chunk of text and the associated original filename.

ann_search(client, query_embedding, window_size=2, top_n=5)
Performs an approximate nearest neighbor (ANN) search using the provided query embedding. Returns the top matching chunks and their descriptions if they are PDF files.

euclidean_search(client, question_embedding)
Performs a Euclidean distance search using the provided question embedding. Returns the most similar chunk of text and the associated original filename.

query_clickhouse_word_with_multi_stage(client, important_words, query_embedding, top_n=5)
Executes a multi-stage search combining keyword matching and semantic similarity. Returns the top matching chunks and descriptions for PDF files.

get_pdf_description(filename)
Retrieves a brief description of a PDF file based on its content from the database. The description is truncated to 300 characters.

deduplicate_results(closest_chunks, top_n)
Removes duplicate results from a list of chunks based on their filenames. Returns unique chunks and their filenames.

structure_sentence_with_llama(query, chunk_text, llama_tokenizer, llama_model)
Structures a given chunk of text in response to a query using the LLaMA model. Returns the structured text.

structure_sentence(query, chunk_text)
Structures a given chunk of text in response to a query using OpenAI's GPT-3.5-turbo model. Returns the structured text.

structure_chunk_text(query, chunk_text)
Formats a given chunk of text without making any changes to its content using OpenAI's GPT-3.5-turbo model. Returns the formatted text.

process_query_clickhouse(query_text, search_method='ann_search')
processes a query by generating embeddings and performing a specified search method in the ClickHouse database. Returns the most relevant chunk of text and its filename.

process_query_clickhouse_pdf(query_text, top_n=5)
Processes a query by extracting important words, generating embeddings, performing a multi-stage search, and retrieving descriptions for the top PDF files. Returns the most relevant contexts, filenames, and descriptions.

search_query.py:
 Imports: Brings in necessary modules and functions for the application.
 
 Flask app initialization: Creates the Flask application instance.
 
 Logging setup: Configures basic logging for the application.
 
 Warning suppression: Ignores specific deprecation warnings.
 
 Conversation history: Initializes an empty list to store chat history.
 
 Main route ('/') handler: Processes GET and POST requests for the main chat interface.
 
 Query processing: Calls functions to process user queries and retrieve responses.
 
 Response formatting: Structures the bot's reply, PDF URLs, and descriptions.
 
 Conversation history update: Adds the latest interaction to the history.
 
 Template rendering: Returns the rendered HTML for the chat interface.
 
 About route ('/about') handler: Renders the About Us page.
 
 Application runner: Starts the Flask application in debug mode when run directly.

Templates
index.html 
 HTML structure: Defines the basic structure of the web page.
 CSS styles: Sets the visual appearance of the chat interface.
 Body background: Uses an image URL for the page background.
 Chat header: Displays the title and navigation buttons at the top.
 Chat container: Holds the main chat interface elements.
 Chat body: Contains the conversation history.
 Message styling: Defines different styles for user and bot messages.
 Chat footer: Provides an input field and send button for user queries.
 Loading animation: Shows a spinner while waiting for responses.
 JavaScript: Handles form submission, API calls, and dynamic content updates.
 Conversation loop: Iterates through conversation history to display messages.
 PDF link display: Shows links to relevant PDF documents with descriptions.
 
about.html 
 HTML structure: Defines the basic structure of the About Us page.
 CSS styles: Sets the visual appearance of the page, including header and content.
 Chat header: Displays the title and navigation buttons at the top.
 Header buttons: Provides links to About Us, Ask Your Question, and Donate pages.
 Content wrapper: Contains the main content of the About Us page.
 About NGO section: Displays information about the organization's mission and focus areas.

requirements.txt
Flask==2.0.1: Web framework for creating the application's interface and handling HTTP requests.
torch==1.9.0: Deep learning library used for neural network operations and tensor computations.
transformers==4.9.2: Provides pre-trained models like BERT for natural language processing tasks.
clickhouse-driver==0.2.0: Client library for interacting with the ClickHouse database.
scipy==1.7.1: Scientific computing library, used here for distance calculations in vector searches.
python-dotenv==0.19.0: Loads environment variables from a .env file for configuration management.
nltk==3.6.2: Natural Language Toolkit for text processing tasks like tokenization.
openai==0.27.0: Client library for interacting with OpenAI's API, used for GPT-3.5 text generation.

PDF Insertion
create_clickhouse_tables: Creates the necessary tables in ClickHouse database if they don't already exist.
 insert_pdf_summary (user_name, original_filename): Inserts a new entry into the 'abc_table' for a PDF summary, returning a unique ID.
extract_text_from_pdf (pdf_path): Extracts text from a PDF file using OCR (Optical Character Recognition).
 insert_chunks (summary_id, pdf_text): Breaks down the PDF text into chunks, generates embeddings, and inserts them into the 'abc_chunks' table.
process_pdf_file (pdf_file_path, user_name): Processes a single PDF file, extracting text, creating a summary, and inserting chunks into the database.
main: The main function that initializes the database tables and processes all PDF files in the specified directory.




