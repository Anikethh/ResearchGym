import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer

# Set GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

# Path to vocab file (expected at repo root)
vocab_file = '../vocab.txt'
output_dir = '../'

# Create a custom tokenizer
tokenizer = BertTokenizer(vocab_file=vocab_file)

# Set maximum sequence length
# NOTE: if you change this, also adjust training/eval scripts accordingly.
max_seq_length = 128

# Define special token IDs
start_tok = tokenizer.convert_tokens_to_ids('[CLS]')
sep_tok = tokenizer.convert_tokens_to_ids('[SEP]')
pad_tok = tokenizer.convert_tokens_to_ids('[PAD]')

def full_sent_tokenize(texts):
    texts = [text.lower() for text in texts]
    tok_sents = [tokenizer(text, padding=False, truncation=False)['input_ids'] for text in tqdm(texts)]
    for s in tok_sents:
        if s and s[0] == start_tok:
            s.pop(0)
    
    res = []
    for s in tok_sents:
        while len(s) > 0:
            chunk = s[:max_seq_length - 2]
            s = s[max_seq_length - 2:]
            chunk = [start_tok] + chunk + [sep_tok]
            if len(chunk) < max_seq_length:
                chunk.extend([pad_tok] * (max_seq_length - len(chunk)))
            res.append(chunk)
    
    attention_mask = [[1 if token != pad_tok else 0 for token in s] for s in res]
    
    return {'input_ids': res, 'attention_mask': attention_mask}

def save_tokenized_output(pt_file, tokenized_output_file):
    data = torch.load(pt_file)
    texts = [entry['text'] for entry in data if 'text' in entry]
    
    tokenized_data = full_sent_tokenize(texts)
    
    # Save tokenized data as .pt file
    torch.save(tokenized_data, tokenized_output_file)
    
    # Reload the saved file for verification
    loaded_data = torch.load(tokenized_output_file)
    print(f"\nTokenized data saved to {tokenized_output_file} and reloaded for verification:\n")
    
    # Print a sample of the saved data
    for i, input_ids in enumerate(loaded_data['input_ids'][:3]):
        print(f"Sample {i + 1} - Token IDs:\n{input_ids}\n")
        print(f"Sample {i + 1} - Tokens:\n{tokenizer.convert_ids_to_tokens(input_ids)}\n")
        print(f"Sample {i + 1} - Decoded Text:\n{tokenizer.decode(input_ids)}\n")

# Set directory paths: process .pt files produced by preprocess.py in this folder
data_dir = '.'
os.makedirs(output_dir, exist_ok=True)

# Process all .pt files in the directory
for file_name in os.listdir(data_dir):
    if file_name.endswith('.pt'):
        pt_file = os.path.join(data_dir, file_name)
        tokenized_output_file = os.path.join(output_dir, f'tokenized_{file_name}')
        
        # Skip if the file has already been processed
        if os.path.exists(tokenized_output_file):
            print(f"File {tokenized_output_file} already exists. Skipping...")
            continue
        
        print(f"Processing file: {pt_file}")
        save_tokenized_output(pt_file, tokenized_output_file)

print("Tokenization complete for all .pt files in the directory.")


