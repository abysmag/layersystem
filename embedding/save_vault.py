import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import torch

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Load VaultGemma-1B
model_name = "google/vaultgemma-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    "google/vaultgemma-1b",
    quantization_config=quantization_config,
    device_map="auto"  # This automatically manages the memory footprint
)

# Ensure proper padding token configuration for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

df = pd.read_csv("cleanedData.csv")
CategoriesList = ["Crime", "Entertainment", "Politics", "Science"]

IDs = df["ID"]
Sentances = df["Content"].tolist()
Categories = df["Category"]

# --- Mini-batch inference to stay within A40 VRAM (48 GB) ---
# Tune BATCH_SIZE down if you still hit OOM; 32 is a safe starting point.
BATCH_SIZE = 32
MAX_LENGTH = 512  # Reduced from 1024; most news sentences are well under 512 tokens.
                  # If your data genuinely needs 1024, set this back and halve BATCH_SIZE.

all_embeddings = []

with torch.no_grad():
    for start in range(0, len(Sentances), BATCH_SIZE):
        batch_texts = Sentances[start : start + BATCH_SIZE]

        # Tokenize only the current batch so padding is batch-local (not global-max)
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to("cuda")

        # output_hidden_states=False: only return the logits + last hidden state
        # via the model's last decoder hidden state, avoiding storing all layer activations.
        outputs = model(
            **inputs,
            output_hidden_states=True,   # still needed to get the embedding
            use_cache=False,             # disable KV-cache; saves additional VRAM
        )

        # Extract ONLY the final layer hidden state, then immediately move to CPU
        # Shape: [batch_size, sequence_length, hidden_dimension]
        last_hidden_state = outputs.hidden_states[-1].cpu().float()

        # Free all other GPU tensors immediately
        del outputs

        # Apply attention-mask mean pooling on CPU to avoid holding large intermediates on GPU
        attention_mask = inputs["attention_mask"].cpu().unsqueeze(-1)  # [B, seq, 1]
        input_mask_expanded = attention_mask.expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        batch_embeddings = (sum_embeddings / sum_mask).numpy()
        all_embeddings.append(batch_embeddings)

        # Explicitly free GPU memory between batches
        del inputs, last_hidden_state, attention_mask, input_mask_expanded
        torch.cuda.empty_cache()

        print(f"Processed {min(start + BATCH_SIZE, len(Sentances))}/{len(Sentances)} documents")

# Final Document Embeddings ready for testing
document_embeddings = np.vstack(all_embeddings)
modelName="Vaultgemma-1b"
np.savez_compressed(
    f"embeddingdata{modelName}.npz",
    embeddings=document_embeddings,
    categories=np.array(Categories),
    texts=np.array(Sentances),
    categorieslist=np.array(CategoriesList),
    embeddingModel=modelName
)

print(f"Saved {document_embeddings.shape[0]} embeddings of dimension {document_embeddings.shape[1]}.")
