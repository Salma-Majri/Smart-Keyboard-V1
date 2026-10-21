import os
import pandas as pd
from datasets import load_dataset

print("1. Loading wikitext dataset from Hugging Face...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

# Extracting train text lines
train_data = dataset['train']['text']

print("2. Cleaning text data...")
# Nettoyage des lignes vides ou des espaces vides
clean_texts = [t.strip() for t in train_data if t.strip() != '']

# Creating a DataFrame
df = pd.DataFrame({'text': clean_texts})

# Assurer que le dossier 'data' existe
output_dir = "tp 2 smart keyboard/02_smart_keyboard_v1/data"
os.makedirs(output_dir, exist_ok=True)

# Saving to corpus.csv
output_path = os.path.join(output_dir, "corpus.csv")
df.to_csv(output_path, index=False)

print(f"3. Success! Dataset successfully saved to {output_path} with {len(df)} lines.")