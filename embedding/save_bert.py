import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("cleanedData.csv")
CategoriesList = ["Crime", "Entertainment", "Politics", "Science"]

#df = df.sample(n=2000, random_state=42).reset_index(drop=True)
IDs = df["ID"]
Sentances = df["Content"]
Categories = df["Category"]

print(device)
# Using standard BERT model for sentence embeddings
model_id = "sentence-transformers/bert-base-nli-mean-tokens"
modelName = model_id.split("/")[-1]
model = SentenceTransformer(model_id).to(device)

# BERT does not require the prompt_name="STS" parameter that Gemma does
embeddings = model.encode(list(Sentances), normalize_embeddings=True)

np.savez_compressed(
    f"embeddingdata{modelName}.npz",
    embeddings=embeddings,
    categories=np.array(Categories),
    texts=np.array(Sentances),
    categorieslist=np.array(CategoriesList),
    embeddingModel=modelName
)

print("saved data")
