from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
# Import the MatryoshkaLoss from Sentence Transformers.
from sentence_transformers.losses import MatryoshkaLoss
import os
import time  # Add time module for execution timing
import traceback

os.environ["WANDB_DISABLED"] = "true"

start_time = time.time()


# For demonstration, we create two examples from a Q&A conversation.
train_examples = [
    InputExample(texts=[
        "Good day everyone and welcome to Harris Corporation second quarter fiscal 2003 earnings release conference call. This call is being recorded; beginning today's meeting is Pamela Padgett, Vice President of Investor Relations. Please go ahead.",
        "Thank you. Hello everyone and welcome to Harris Corporation second quarter fiscal 2003 conference call. I am Pamela Padgett, Vice President Investor Relations and with me on the call today is Mr. Phillip Farmer, our Chairman and CEO, Mr. Bryan Roub, our Chief Financial Officer."
    ], label=1.0)
]


train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Initialize a base model
model = SentenceTransformer("/project/aaz/leo/models/all-mpnet-base-v2")

# Define a base loss (you may experiment with MultipleNegativesRankingLoss or CosineSimilarityLoss)
base_loss = losses.CosineSimilarityLoss(model=model)

# Define the target dimensions for the nested (Matryoshka) embeddings.
# These tell the model that the first 768, 512, 256, 128, and 64 dimensions should each be trained to be useful.
matryoshka_dims = [768, 512, 256, 128, 64]
matryoshka_weights = [1] * len(matryoshka_dims)



# Wrap the base loss with MatryoshkaLoss.
matryoshka_loss = MatryoshkaLoss(model=model,
                                 loss=base_loss,
                                 matryoshka_dims=matryoshka_dims,
                                 matryoshka_weights=matryoshka_weights)

# Create output directory if it doesn't exist
output_path = "outputs/finetuned-matryoshka-modelv1"
os.makedirs(output_path, exist_ok=True)
print("-------------Start Training------------")
# Calculate and print execution time even if there was an error


try:
    # Fine-tune the model using the training DataLoader and MatryoshkaLoss.
    # Adjust the number of epochs and warmup steps according to your data and compute capacity.
    model.fit(train_objectives=[(train_dataloader, matryoshka_loss)],
            epochs=30,
            warmup_steps=10,
            output_path=output_path)
    
    print("-------------Training Completed------------")
    execution_time = time.time() - start_time
    print(f"Execution time before error: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    # Load the fine-tuned model and specify a target dimension at inference.
    try:
        model = SentenceTransformer(output_path, truncate_dim=128)
        print(f"Successfully loaded model from {output_path}")
        
        # Example sentences (you can use any text; here we use parts of our conversation)
        sentences = [
            "Good day everyone and welcome to Harris Corporation second quarter fiscal 2003 earnings release conference call.",
            "Thank you. Hello everyone and welcome to Harris Corporation conference call."
        ]

        # Get the embeddings (each embedding will be 128-dimensional)
        embeddings = model.encode(sentences)
        print("Embeddings shape:", embeddings.shape)  # Expected output: (number of sentences, 128)

        print("-------------Mission Completed------------")
    except Exception as e:
        print(f"Error during inference: {str(e)}") 
        traceback.print_exc()
        
except Exception as e:
    print(f"Error during training: {str(e)}")
    traceback.print_exc()

# Calculate and print execution time even if there was an error
execution_time = time.time() - start_time
print(f"Execution time before error: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")