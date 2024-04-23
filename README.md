# Project Overview

This project provides an automated way to perform Google searches, process the HTML content of the resulting pages, and use natural language processing to generate answers to user queries based on the search results. The application utilizes several libraries including Playwright, BeautifulSoup, LangChain, and Hugging Face's transformers.

## Code Functionality

### 1. **Google Search**

- **Function**: `google_search(query)`
- **Purpose**: Performs a Google search using the provided query and collects the URLs of the search results using requests and BeautifulSoup.
- **Details**: This function sends a request to Google's search page, parses the resulting HTML to extract the main result links, and returns a list of these URLs.

### 2. **Load HTML Documents**

- **Components Used**: `AsyncChromiumLoader`
- **Purpose**: Loads the full HTML content of the web pages obtained from the Google search results.
- **Details**: Utilizes Playwright operated through an asynchronous Chromium browser instance to fetch and return the HTML content of each URL.

### 3. **Extract Content from HTML**

- **Components Used**: `BeautifulSoupTransformer`
- **Purpose**: Extracts relevant textual content from the HTML documents.
- **Details**: Processes the loaded HTML documents to extract text specifically from tags like `<p>`, `<li>`, `<div>`, and `<a>`, which commonly contain informative content.

### 4. **Split Text Documents**

- **Components Used**: `RecursiveCharacterTextSplitter`
- **Purpose**: Splits the extracted text into smaller chunks.
- **Details**: Divides large text blocks into smaller, more manageable sizes using specified parameters for chunk size and overlap, enhancing processing efficiency in subsequent steps.

### 5. **Vector Database Creation**

- **Components Used**: `Chroma`, `HuggingFaceEmbeddings`
- **Purpose**: Creates a persistent vector database of the text chunks.
- **Details**: Converts text chunks into vector embeddings using pretrained models from Hugging Face, and stores these vectors in a ChromaDB database for efficient retrieval.

### 6. **Retrieval and Answering by LLM**

- **Components Used**: `ConversationalRetrievalChain`, `Ollama`
- **Purpose**: Retrieves relevant text chunks based on the user's query and generates answers.
- **Details**: Uses a vector-based retriever to find the most relevant text chunks for a new query and an LLM (like LLaMA) to generate coherent and contextually relevant answers.

### 7. **Interactive Query-Answer System**

- **Details**: The system prompts the user for queries, uses the retrieval and answering components to find and generate responses, and displays these to the user, allowing for interactive dialogue.

## Running the Application

To run the application, ensure all dependencies are installed and execute:
