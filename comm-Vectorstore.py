import json
import requests
import random

# Load embeddings and questions from JSON files
with open('/Users/abhigyan/Downloads/project/RAG + LLM/em_100.json', 'r') as embeddings_file:
    embeddings_data = json.load(embeddings_file)

with open('/Users/abhigyan/Downloads/project/VODB/vodb/question.json', 'r') as questions_file:
    questions_data = json.load(questions_file)

def insert_all_embeddings(tree_name):
    """Insert all embeddings into the specified tree individually."""
    for item in embeddings_data:
        chunk = item.get("chunk")
        embedding = item.get("embedding")
        
        # Prepare the payload for the request to insert
        insert_payload = {
            "data": chunk,
            "embedding": embedding,
        }

        # Send a POST request to insert the point into the specified tree
        insert_response = requests.post(f'http://127.0.0.1:8080/insert?tree_name={tree_name}', json=insert_payload)

        print(f'Sent for insertion: {insert_payload}')
        print(f'Insertion Response: {insert_response.status_code}')
        
        if insert_response.status_code != 200:
            print(f'Error inserting point: {insert_response.text}')
            break  # Stop inserting if there's an error

def bulk_insert_embeddings(tree_name):
    """Insert all embeddings into the specified tree in bulk."""
    # Prepare the bulk insert payload
    bulk_insert_payload = {"points": []}  # Use 'points' instead of 'embeddings'
    
    for item in embeddings_data:
        chunk = item.get("chunk")
        embedding = item.get("embedding")
        bulk_insert_payload["points"].append({
            "data": chunk,
            "embedding": embedding,
        })

    # Send a POST request to bulk insert the points into the specified tree
    bulk_insert_response = requests.post(f'http://127.0.0.1:8080/bulk_insert?tree_name={tree_name}', json=bulk_insert_payload)

    print(f'Sent for bulk insertion: {bulk_insert_payload}')
    print(f'Bulk Insertion Response: {bulk_insert_response.status_code}')
    
    if bulk_insert_response.status_code != 200:
        print(f'Error bulk inserting points: {bulk_insert_response.text}')


def retrieve_nearest_neighbors(tree_name, n_neighbors=5):
    """Retrieve nearest neighbors for a randomly selected embedding."""
    # Randomly select an embedding
    random_item = random.choice(embeddings_data)
    chunk = random_item.get("chunk")
    embedding = random_item.get("embedding")

    # Prepare the payload for the nearest neighbors request
    nearest_neighbors_payload = {
        "data": chunk,
        "embedding": embedding,
    }

    # Send a POST request to get the nearest neighbors
    neighbor_response = requests.post(f'http://127.0.0.1:8080/nearesttop?n={n_neighbors}&tree_name={tree_name}', json=nearest_neighbors_payload)

    print(f'Nearest Neighbor Response: {neighbor_response.status_code}')
    
    if neighbor_response.status_code == 200:
        nearest_neighbors = neighbor_response.json()
        print(f'Found {len(nearest_neighbors)} nearest neighbors:')
        for neighbor in nearest_neighbors:
            print(neighbor)
    else:
        print(f'Error retrieving nearest neighbors: {neighbor_response.text}')

if __name__ == '__main__':
    tree_name = "em1000"
    
    # Menu for user options
    print("Select an option:")
    print("1: Insert all embeddings")
    print("2: Bulk Insert all embeddings")  # Option for bulk insert
    print("3: Retrieve nearest neighbors")

    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == '1':
        insert_all_embeddings(tree_name)
    elif choice == '2':
        bulk_insert_embeddings(tree_name)  # Call the new bulk insert function
    elif choice == '3':
        n_neighbors = int(input("Enter the number of nearest neighbors you want to retrieve: "))
        retrieve_nearest_neighbors(tree_name, n_neighbors)
    else:
        print("Invalid choice. Please select 1, 2, or 3.")
