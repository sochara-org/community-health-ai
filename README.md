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

### functions.py
**initialize_clickhouse_connection()**
Initializes a connection to the ClickHouse database using the credentials and configurations stored in environment variables.

**initialize_tokenizer_and_model()**
Loads a pre-trained BERT tokenizer and model from the Hugging Face library. If the tokenizer lacks a padding token, it assigns a default one.

**initialize_llama_model()**
Loads a pre-trained LLaMA tokenizer and model from the Hugging Face library for causal language modeling tasks.

**generate_embeddings(tokenizer, model, query)**
Generates embeddings for a given query text using the provided tokenizer and model. Returns the pooled embedding vector.

**extract_important_words(query_text)**
Extracts important words from a given query text, excluding common stop words. Returns a list of significant words.

**get_surrounding_chunks(client, id, summary_id, window_size=2)**
Fetches chunks of text surrounding a specific chunk from the database. The window_size parameter determines how many chunks before and after should be included.

**get_original_filename(client, summary_id)**
Retrieves the original filename associated with a given summary ID from the database, processes it, and constructs the full file URL.

**cosine_similarity(client, question_embedding)**
Performs a cosine similarity search using the provided question embedding. Returns the most similar chunk of text and the associated original filename.

**vector_search_cosine_distance(client, question_embedding)**
Performs a cosine distance search using the provided question embedding. Returns the most similar chunk of text and the associated original filename.

**ann_search(client, query_embedding, window_size=2, top_n=5)**
Performs an approximate nearest neighbor (ANN) search using the provided query embedding. Returns the top matching chunks and their descriptions if they are PDF files.

**euclidean_search(client, question_embedding)**
Performs a Euclidean distance search using the provided question embedding. Returns the most similar chunk of text and the associated original filename.

**query_clickhouse_word_with_multi_stage(client, important_words, query_embedding, top_n=5)**
Executes a multi-stage search combining keyword matching and semantic similarity. Returns the top matching chunks and descriptions for PDF files.

**get_pdf_description(filename)**
Retrieves a brief description of a PDF file based on its content from the database. The description is truncated to 300 characters.

**deduplicate_results(closest_chunks, top_n)**
Removes duplicate results from a list of chunks based on their filenames. Returns unique chunks and their filenames.

**structure_sentence_with_llama(query, chunk_text, llama_tokenizer, llama_model)**
Structures a given chunk of text in response to a query using the LLaMA model. Returns the structured text.

**structure_chunk_text(query, chunk_text)**
Formats a given chunk of text without making any changes to its content using OpenAI's GPT-3.5-turbo model. Returns the formatted text.

**process_query_clickhouse(query_text, search_method='ann_search')**
processes a query by generating embeddings and performing a specified search method in the ClickHouse database. Returns the most relevant chunk of text and its filename.

**process_query_clickhouse_pdf(query_text, top_n=5)**
Processes a query by extracting important words, generating embeddings, performing a multi-stage search, and retrieving descriptions for the top PDF files. Returns the most relevant contexts, filenames, and descriptions.

### search_query.py

This file contains the main Flask application for the chat interface.

#### Key Components:
**Imports** Imports necessary modules and functions, including Flask and custom functions.

**Flask App Initialization**Creates the Flask application instance.

**Logging and Warning Configuration** Sets up basic logging and suppresses specific deprecation warnings.

**Conversation History** Initializes an empty list to store the chat history.

**Main Route ('/')**
   - Handles both GET and POST requests for the main chat interface.
   - POST request processing:
     - Retrieves the user's query.
     - Calls `process_query_clickhouse_pdf` to get responses and PDF information.
     - Formats the response with bot reply, PDF URLs, and descriptions.
     - Updates the conversation history.
   - GET request:
     - Renders the main chat interface template.

**About Route ('/about')** Renders the About Us page.

**Error Handling** Provides basic error responses for invalid queries or processing errors.

**Application Runner** Starts the Flask application in debug mode when the script is run directly.

#### Key Functions:

- `index()`: Main route handler for the chat interface.
- `about()`: Handler for the About page.

#### Templates Used:

- `index.html`: Main chat interface template.
- `about.html`: About page template.

This script sets up a web application that allows users to interact with a chat interface, process queries, and receive responses along with relevant PDF information. It maintains a conversation history and provides a simple About page.

### Templates
### index.html 
## HTML Structure
The HTML structure is divided into several main sections:

 **Header**
    - Contains the title of the chatbot and navigation buttons.
 **Chat Container**
    - Encloses the chat header, chat body, and chat footer.
 **Chat Body**
    - Displays the conversation history between the user and the bot.
 **Chat Footer**
    - Contains the input form for users to type and send their messages.
 **Loading Animation**
    - Displays a spinner animation while the chatbot processes the user query.

## CSS Styling
The CSS styles enhance the appearance and layout of the interface.

### Key Styles:
 **Body**
    - Background image, font settings, and overall layout.
 **Chat Header**
    - Fixed position with background color and text alignment.
 **Header Buttons**
    - Style for the navigation buttons with hover effects.
 **Chat Container and Body**
    - Layout settings for chat display, including flexbox settings.
 **Messages**
    - Styles for user and bot messages with different background colors.
 **Loading Animation**
    - Spinner animation to indicate processing.

## JavaScript Functionality
The JavaScript handles user interaction and communication with the server.

### Key Functions:
 **Scroll to Bottom**
    - Ensures the chat scrolls to the latest message.
 **Form Submission**
    - Handles user input and sends it to the server using XMLHttpRequest.
 **Loading Animation**
    - Displays a spinner while the server processes the query.
 **Response Handling**
    - Updates the chat body with the user's message, bot's response, and any PDF links and descriptions.
 
### about.html 
 ## Overview
The About Us page provides information about the organization, its mission, and focus areas. The page includes a header with navigation buttons, and a content section with details about the NGO.

## HTML Structure
The HTML structure is divided into several main sections:

1. **Document Type and Language**
    - Defines the document type and language of the page.
2. **Head Section**
    - Contains meta information and links to stylesheets.
3. **Body Section**
    - Encloses the chat header and content wrapper.



### requirements.txt
- **Flask**: Web framework for creating the application's interface and handling HTTP requests.
- **torch**: Deep learning library used for neural network operations and tensor computations.
- **transformers**: Provides pre-trained models like BERT for natural language processing tasks.
- **clickhouse-driver**: Client library for interacting with the ClickHouse database.
- **scipy**: Scientific computing library, used here for distance calculations in vector searches.
- **python-dotenv**: Loads environment variables from a .env file for configuration management.
- **nltk**: Natural Language Toolkit for text processing tasks like tokenization.
- **openai**: Client library for interacting with OpenAI's API, used for GPT-3.5 text generation.
- **PyPDF2**: Library for reading and extracting text from PDF files.
- **uuid**: Generates unique identifiers for database entries.
- **pytesseract**: OCR tool for extracting text from images (used for scanned PDFs).
- **pdf2image**: Converts PDF pages to images for OCR processing.


### pdf_uploading.py
**create_clickhouse_tables()** Sets up necessary tables in ClickHouse.

**insert_pdf_summary()** Inserts a new entry for a PDF file into the database.

**extract_text_from_pdf()** Extracts text from PDF files, including OCR for scanned documents.

**insert_chunks()** Splits PDF text into chunks, generates embeddings, and inserts into the database.

**process_pdf_file()** Orchestrates the processing of a single PDF file.

**main()** Processes all PDF files in a specified directory.


### pdf_downloading.py

This script automates the process of downloading PDF files based on metadata retrieved from a specified API.

### Key Components:

**Environment Setup** Uses `dotenv` to load environment variables from a `.env` file.
 - Retrieves crucial URLs and directory paths from environment variables:
     - `METADATA_URL`: API endpoint for metadata
     - `OUTPUT_DIRECTORY`: Where PDFs will be saved
     - `ARCHIVE_BASE_URL`: Base URL for constructing PDF download links

**Main Functions**

**download_pdfs_from_metadata(metadata_url, output_dir)**
      - Fetches metadata from the specified API
      - Extracts PDF identifiers from the metadata
      - Constructs individual PDF URLs
      - Initiates download for each PDF

**download_pdf(pdf_url, pdf_filename)**
      - Downloads a single PDF file from the given URL
      - Saves the PDF to the specified output directory
      - Prints confirmation message upon successful download

**Execution Flow**
   - Creates the output directory if it doesn't exist
   - Calls `download_pdfs_from_metadata()` to start the download process






