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
    device_map="auto" # This automatically manages the memory footprint
)

# Ensure proper padding token configuration for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

df = pd.read_csv("cleanedData.csv")
CategoriesList = ["Crime", "Entertainment", "Politics", "Science"]

IDs = df["ID"]
Sentances = df["Content"].tolist()
Categories = df["Category"]

inputs = tokenizer(Sentances, padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Extract the final layer hidden states
# Shape: [batch_size, sequence_length, hidden_dimension]
last_hidden_state = outputs.hidden_states[-1] 

# Apply Attention Mask Mean Pooling for precise document vectors
attention_mask = inputs['attention_mask'].unsqueeze(-1) # [batch_size, seq_len, 1]
input_mask_expanded = attention_mask.expand(last_hidden_state.size()).float()
sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
sum_mask = input_mask_expanded.sum(1)
sum_mask = torch.clamp(sum_mask, min=1e-9)

# Final Document Embeddings ready for testing
document_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

np.savez_compressed(
    "embeddingdataVault.npz",
    embeddings=document_embeddings,
    categories=np.array(Categories),
    texts=np.array(Sentances),
    categorieslist=np.array(CategoriesList)
)
