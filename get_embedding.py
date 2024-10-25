import os
import json
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk

# Preprocess text by loading and splitting into sentences
def load_text_file(file_path):
    nltk.download('punkt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Use NLTK to split into sentences
    sentences = sent_tokenize(text)
    return sentences

# Generate embeddings for each sentence using a pre-trained model
def generate_embeddings(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings, model

# Chunk sentences based on a predefined maximum length
def chunk_text_by_length(sentences, max_length=2000):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        # If adding this sentence exceeds max_length, start a new chunk
        if current_length + sentence_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Append the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    print(f"Created {len(chunks)} chunks based on length.")
    return chunks

# Store chunks and their embeddings in a JSON file
def store_chunks_in_json(chunks, embeddings, output_file):
    chunks_data = [{"chunk": chunk, "embedding": embedding.tolist()} for chunk, embedding in zip(chunks, embeddings)]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=4)

    print(f"Stored {len(chunks)} chunks in '{output_file}'.")

# Main function to load the text, process it, and store chunks with embeddings
def main(file_path, output_file):
    # Load and preprocess text
    sentences = load_text_file(file_path)

    # Generate sentence embeddings and initialize model
    sentence_embeddings, model = generate_embeddings(sentences)
    print(f"Generated embeddings for {len(sentences)} sentences.")

    # Chunk the text based on length
    chunks = chunk_text_by_length(sentences, max_length=500)  # Adjust max_length as needed

    # Generate embeddings for the chunks
    chunk_embeddings = model.encode(chunks)

    # Store chunks and embeddings in a JSON file
    store_chunks_in_json(chunks, chunk_embeddings, output_file)

if __name__ == "__main__":
    file_path = input("Enter the path to your .txt file: ")
    output_file = input("Enter the path for the output JSON file: ")
    main(file_path, output_file)
