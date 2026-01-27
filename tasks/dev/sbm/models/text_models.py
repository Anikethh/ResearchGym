from transformers import AlbertForSequenceClassification
from transformers import BertForSequenceClassification
from transformers import DebertaV2ForSequenceClassification
from transformers import BertModel
import types
import torch
import torch.nn as nn
from .register import register_model

class BertFeatureWrapper(torch.nn.Module):

    def __init__(self, model, use_relu=False):
        super().__init__()
        self.model = model
        self.num_features = model.config.hidden_size
        classifier_dropout = model.config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        if use_relu:
            self.act = nn.ReLU()
        self.use_relu = use_relu
    def forward(self, x):
        kwargs = {
            'input_ids': x[:, :, 0],
            'attention_mask': x[:, :, 1]
        }
        if x.shape[-1] == 3:
            kwargs['token_type_ids'] = x[:, :, 2]
        output = self.model(**kwargs)
        if self.use_relu:
            if hasattr(output, 'pooler_output'):
                return self.act(self.dropout(output.pooler_output))
            else:
                return self.act(self.dropout(output.last_hidden_state[:, 0, :]))
        else:
            if hasattr(output, 'pooler_output'):
                return self.dropout(output.pooler_output)
            else:
                return self.dropout(output.last_hidden_state[:, 0, :])

def _bert_replace_fc(model):
    model.fc = model.classifier
    delattr(model, "classifier")

    def classifier(self, x):
        return self.fc(x)
    
    model.classifier = types.MethodType(classifier, model)

    model.base_forward = model.forward

    def forward(self, x):
        return self.base_forward(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2]).logits

    model.forward = types.MethodType(forward, model)
    return model

@register_model("bert-base-uncased")
def bert_feature(pretrained, use_relu=True):
    if pretrained:
        text_model = BertModel.from_pretrained('bert-base-uncased')
    else:
        config_class = BertModel.config_class
        config = config_class.from_pretrained('bert-base-uncased')
        text_model = BertModel(config)
    return BertFeatureWrapper(text_model, use_relu)

def bert_pretrained(output_dim):
	return _bert_replace_fc(BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=output_dim))


def bert_pretrained_multilingual(output_dim):
    return _bert_replace_fc(BertForSequenceClassification.from_pretrained(
            'bert-base-multilingual-uncased', num_labels=output_dim))


def bert(output_dim):
    config_class = BertForSequenceClassification.config_class
    config = config_class.from_pretrained(
            'bert-base-uncased', num_labels=output_dim)
    return _bert_replace_fc(BertForSequenceClassification(config))


def bert_large_pretrained(output_dim):
    return _bert_replace_fc(BertForSequenceClassification.from_pretrained(
            'bert-large-uncased', num_labels=output_dim))


def deberta_pretrained(output_dim):
    return _bert_replace_fc(DebertaV2ForSequenceClassification.from_pretrained(
            'microsoft/deberta-v3-base', num_labels=output_dim))


def deberta_large_pretrained(output_dim):
    return _bert_replace_fc(DebertaV2ForSequenceClassification.from_pretrained(
            'microsoft/deberta-v3-large', num_labels=output_dim))


def albert_pretrained(output_dim):
    return _bert_replace_fc(AlbertForSequenceClassification.from_pretrained(
            'albert-base-v2', num_labels=3))