# RAG-Framework

A Python framework to implement Retrieval-Augmented Generation (RAG) using a Vector-store Database. This framework utilizes the [Vector-Store](https://github.com/Abhigyan126/Vector-Store) repository to store and retrieve embeddings for effective RAG implementation.

## Structure

- **main.py** - The main RAG pipeline framework, processing input and orchestrating the RAG flow.
- **llm.py** - A module that interfaces with the Google Gemini API, simulating a Large Language Model (LLM) for generating responses.
- **get_embedding.py** - A script for generating embeddings from `.txt` files using the SentenceTransformer model.
- **comm-Vectorstore.py** - A communication module that uploads embeddings to the Vector Store for storage and retrieval.

## Usage

1. Clone the [Vector-Store](https://github.com/Abhigyan126/Vector-Store) repository and set it up according to the instructions.
2. Configure the `main.py` and `get_embedding.py` to use your Vector-store Database.
3. Run `main.py` to execute the RAG pipeline with any input text.

## Requirements

- Python 3.x
- [Sentence-Transformers](https://www.sbert.net/) - For generating text embeddings
- [nltk](https://www.nltk.org/) - For sentence tokenization
- Google Gemini API credentials for `llm.py` (or any compatible LLM API for inference)

## Installation

```bash
pip install -r requirements.txt
```
