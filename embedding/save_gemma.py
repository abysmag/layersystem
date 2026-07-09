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

        

#print(Categories)
print(device)
model_id = "google/embeddinggemma-300m"
modelName = model_id.split("/")[-1]
model = SentenceTransformer(model_id).to(device)
embeddings = model.encode(list(Sentances), prompt_name="STS", normalize_embeddings=True)


#for idx, embedding in enumerate(embeddings):
    #print(f"Embedding {idx+1} shape: {embedding.shape}")
    #print(embedding)

np.savez_compressed(
    f"embeddingdata{modelName}.npz",
    embeddings=embeddings,
    categories=np.array(Categories),
    texts=np.array(Sentances),
    categorieslist=np.array(CategoriesList),
    embeddingModel=modelName
)

print("saved data")
