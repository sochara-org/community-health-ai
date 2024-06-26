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
initialize_clickhouse_connection: Establishes a ClickHouse database connection.
initialize_tokenizer_and_model: Initializes a tokenizer and model for generating embeddings.
generate_embeddings: Generates embeddings for a given query text.
rank_sections: Ranks text sections based on cosine similarity to the query.
cosine_similarity: Retrieves the most similar section using cosine similarity.
vector_search_cosine_distance: Retrieves the most similar section using cosine distance.
ann_search: Retrieves the most similar section using approximate nearest neighbors search.
euclidean_search: Retrieves the most similar section using Euclidean distance.
extract_important_words: Extracts important words from the query text.
ann_search_on_chunks: Performs approximate nearest neighbors search on chunks containing important words.
get_surrounding_chunks: Retrieves surrounding chunks for additional context.
structure_sentence: Structures text using the GPT-3.5 model.
structure_chunk_text: Formats text without adding or removing content.
process_query: Processes a query and returns a structured response.
process_query_clickhouse: Processes a query using different search methods and returns a structured response.
process_query_clickhouse_word: Processes a query by extracting important words and retrieving surrounding context.

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




