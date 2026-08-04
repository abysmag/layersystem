import argparse

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from dataset_config import add_dataset_args, get_dataset_config

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Generate Qwen embeddings for a dataset.")
add_dataset_args(parser)
args = parser.parse_args()

cfg = get_dataset_config(args.dataset)
df = pd.read_csv(cfg["csv"])
CategoriesList = cfg["categories"]

#df = df.sample(n=2000, random_state=42).reset_index(drop=True)
IDs = df[cfg["id_col"]]
Sentances = df[cfg["text_col"]]
Categories = df[cfg["category_col"]]

        

#print(Categories)
print(device)
model_id = "Qwen/Qwen3-Embedding-8B"
modelName = model_id.split("/")[-1]
model = SentenceTransformer(model_id, truncate_dim=768).to(device)
embeddings = model.encode(list(Sentances), normalize_embeddings=True, truncate_dim=768)



#for idx, embedding in enumerate(embeddings):
    #print(f"Embedding {idx+1} shape: {embedding.shape}")
    #print(embedding)

np.savez_compressed(
    f"embeddingdata{modelName}_{args.dataset}.npz",
    embeddings=embeddings,
    categories=np.array(Categories),
    texts=np.array(Sentances),
    categorieslist=np.array(CategoriesList),
    embeddingModel=modelName,
    dataset=args.dataset
)

print(f"saved data for dataset '{args.dataset}'")
