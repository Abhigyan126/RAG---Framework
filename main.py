from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from llm import LLM
import requests
import nltk
import re

# Begin

class RAG:
    def __init__(self):
        # -Load
        self.llm = LLM()
        nltk.download('punkt')
        nltk.download('punkt_tab')
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 

    # -Functions
    # --Func to generate embedding from text
    def generate_embeddings_json(self, sentence):
        sentences = sent_tokenize(sentence)  # tokenize sentence
        sentence_embeddings = self.model.encode(sentences)  # generate embedding
        if sentences:
            chunks_data = {
                "data": sentences[0], 
                "embedding": sentence_embeddings[0].tolist()
            }
        else:
            chunks_data = {"data": "", "embedding": []}

        return chunks_data

    # --Func to clean data
    def clean_text(self, sentences):
        try:
            cleaned_sentences = str()
            for sentence in sentences:
                cleaned = re.sub(r'\[.*?\]', '', sentence)
                cleaned = cleaned.replace('\n', ' ')
                cleaned = cleaned.strip()
                if cleaned:
                    cleaned_sentences += ". " + cleaned

            return cleaned_sentences
        except Exception as e:
            print("Error :", e)

    # --Func to fetch nearest neighbour from DB
    def get_nearest_embeddings(self, json_data, tree_name, n=5):
        data = json_data['data']
        e = json_data['embedding']
        nearest_neighbors_payload = {
            "data": data,
            "embedding": e,
        }
        neighbor_response = requests.post(f'http://127.0.0.1:8080/nearesttop?n={n}&tree_name={tree_name}', json=nearest_neighbors_payload)
        if neighbor_response.status_code == 200:
            nearest_neighbors = neighbor_response.json()
            print(f'Found {len(nearest_neighbors)} nearest neighbors:')
            return nearest_neighbors
        else:
            return neighbor_response.status_code

    # -Func to Pipeline
    def pipeline(self, message):
        embeded_json = self.generate_embeddings_json(message)
        requested_json = self.get_nearest_embeddings(embeded_json, 'random_tree', 5)
        nearest_embedding = [item['data'] for item in requested_json]
        text = self.clean_text(nearest_embedding)
        text = f'you are RAG bot your job is to use data provided and give natural response add nothing outside the data provided, ignore providing information by yourself.answer only relevent information using the question:{message} from the provided data and ignore non relevent. data: {text}'
        text = self.llm.model(text)
        return text

# -Main
if __name__ == "__main__":
    processor = RAG()
    message = input("Enter the message you want to process: ")
    print(processor.pipeline(message))
