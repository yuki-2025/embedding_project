# finetune/pca.py
import os
import numpy as np
import torch
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_results(metadata_file):
    """
    Loads data from a pickle metadata file and associated embedding file.

    Args:
        metadata_file (str): Path to the pickle file containing metadata.

    Returns:
        list: A list of dictionaries, each containing metadata and embedding for an example.
    """
    # Load metadata
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    
    # All examples reference the same embedding file in this case
    embedding_file = metadata[0]["embedding_file"]
    all_embeddings = np.load(embedding_file)
    
    loaded_data = []
    for i, entry in enumerate(metadata):
        example_id = entry["example_id"]
        input_text = entry["input"]
        generated_text = entry["generated_text"]
        rouge_scores = entry["rouge_scores"]
        
        # Get the corresponding embedding for this example (one row per example)
        embedding = torch.from_numpy(all_embeddings[i:i+1])
        # embedding = all_embeddings[i]  # Corrected: Access directly, it's already a NumPy array
        
        loaded_data.append({
            "example_id": example_id,
            "input": input_text,
            "generated_text": generated_text,
            "embedding": embedding,
            "rouge_scores": rouge_scores,
        })

    return loaded_data

def visualize_embeddings(loaded_data):
    """
    Apply PCA to the embeddings and visualize them.
    """
    # Extract embeddings and create labels
    embeddings = torch.cat([data["embedding"] for data in loaded_data], dim=0).numpy()
    labels = [f"Example {data['example_id']}" for data in loaded_data]
    
    # Perform PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, marker='o', s=100)
        plt.text(x, y, labels[i], fontsize=9)
    
    plt.title('PCA of Model Embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    output_dir = os.path.dirname(metadata_file)
    plt.savefig(os.path.join(output_dir, 'embeddings_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA visualization saved to {os.path.join(output_dir, 'embeddings_pca.png')}")

# Example usage:
metadata_file = "./results/embedding/metadata.pkl"
loaded_results = load_results(metadata_file)

# Loop and print all results:
for i, result in enumerate(loaded_results):
    print(f"\n--- Example {i + 1} ---")
    print(f"  Input: {result['input'][:1000]}...")  # Show beginning of input
    print(f"  Generated text: {result['generated_text'][:1000]}...")  # Show beginning of generated text
    print(f"  Embedding shape: {result['embedding'].shape}")
    
    # Print ROUGE scores
    print(f"  ROUGE Scores:")
    for metric, scores in result['rouge_scores'].items():
        print(f"    {metric}:")
        for score_type, value in scores.items():
            print(f"      {score_type}: {value:.4f}")

# Visualize the embeddings
visualize_embeddings(loaded_results)

# Additional: Calculate cosine similarities between embeddings
if len(loaded_results) > 1:
    print("\n--- Cosine Similarities Between Embeddings ---")
    for i in range(len(loaded_results)):
        for j in range(i+1, len(loaded_results)):
            emb1 = loaded_results[i]["embedding"]
            emb2 = loaded_results[j]["embedding"]
            # emb1 = torch.tensor(loaded_results[i]["embedding"]).unsqueeze(0)  # Convert to tensor *here*
            # emb2 = torch.tensor(loaded_results[j]["embedding"]).unsqueeze(0)  # and add batch dimension
            
            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
            print(f"Similarity between Example {i+1} and Example {j+1}: {cos_sim:.4f}")