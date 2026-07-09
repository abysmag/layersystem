import pandas as pd

df = pd.read_csv("combinedog.csv")
CategoriesList = ["Crime", "Entertainment", "Politics", "Science"]

# Drop rows where 'Content' is missing (NaN) or not a string
df = df[df["Content"].apply(lambda x: isinstance(x, str))]
df = df.reset_index(drop=True)

#df = df.sample(n=2000, random_state=42).reset_index(drop=True)
IDs = df["ID"]
Sentances = df["Content"]
Categories = df["Category"]

df.to_csv('cleanedData.csv', index=False)