import random
from tqdm import tqdm
from argparse import ArgumentParser
from tokenizers.normalizers import BertNormalizer
import torch

num = 8
pt_file = f'./data_batch_{num}.pt'
output_pt_file = f'preprocessed_MatSciBERT_data_{num}.pt'

# Load mappings
with open('vocab_mappings.txt') as f:
    mappings = f.read().strip().split('\n')

mappings = {m[0]: m[2:] for m in mappings}

# Initialize normalizer
norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize_text(text):
    normalized_text = norm.normalize_str(text)
    norm_sent = ""
    for c in normalized_text:
        if c in mappings:
            norm_sent += mappings[c]
        elif random.uniform(0, 1) < 0.3:
            norm_sent += c
        else:
            norm_sent += ' '
    return norm_sent

# Load PT file
pt_data = torch.load(pt_file)

# Initialize an empty list to hold normalized data
normalized_data = []

# Process each item and save immediately
with open(output_pt_file, 'wb') as f:
    for item in tqdm(pt_data):
        # Copy the item to ensure the original structure is maintained
        processed_item = item.copy()
        # Normalize the text field
        processed_item['text'] = normalize_text(item['text'])
        # Append the processed item to the list
        normalized_data.append(processed_item)
        
        # Save the processed data immediately to avoid memory issues
        torch.save(normalized_data, f)


