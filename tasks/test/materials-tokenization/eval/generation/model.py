import pandas as pd
import numpy as np
import torch, json, os
from tqdm import tqdm
from datetime import datetime,timedelta
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


class LanguageModel(nn.Module):
    def __init__(self, params):
        super(LanguageModel, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.params = params
        self.encoder = params['encoder'].to(self.device)
        self.decoder = params['decoder'].to(self.device)
        self.linear  = nn.Linear(params['hidden_dim'],params['vocab_size']).to(self.device)

        self.tgt_mask = self.generate_square_subsequent_mask(params['max_seq_length']-1).to(self.device)

    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, batch):
        # encoder
        input_ids = batch['inputs']['input_ids'].to(self.device)
        attention_mask = batch['inputs']['attention_mask'].to(self.device)
        hidden_emb = self.encoder(input_ids,attention_mask)[0]
        # decoder
        indices = torch.LongTensor(batch['answers']['input_ids']).to(self.device)
        tgt = self.encoder.embeddings.word_embeddings(indices).to(self.device)
        output = self.linear(self.decoder(tgt,hidden_emb,self.tgt_mask))
        return output