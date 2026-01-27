import pandas as pd
import numpy as np
import torch, json, os
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import datetime,timedelta
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import random
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    AutoConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)
from transformers import BatchEncoding

question_type_dict = {'ner':0,'pc':1,'rc':2,'ee':3,'sar':4,'sc':5,'sf':6}
default_sample_num = 2
dir_path = '../dataset'

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data     = dataset[0]
        self.answer   = dataset[1]
        self.qtype    = dataset[2] 

    def __getitem__(self, idx):
        data = dict()
        data['inputs']    = {key: val[idx].clone().detach() for key, val in self.data.items()}
        data['answers']   = {key: val[idx].clone().detach() for key, val in self.answer.items()}
        data['qtype']     = self.qtype[idx]
        return data

    def __len__(self):
        return len(self.data.input_ids)


def tokenize_bpe(df, max_length=512, tokenizer=None):
    texts = df['texts'].values.tolist()
    questions = df['questions'].values.tolist()
    answers = df['answers'].values.tolist()
    qtypes = df['qtypes'].values.tolist()

    # Initialize dictionaries for input_ids and attention_mask
    data_tokens = {
        'input_ids': [],
        'attention_mask': []
    }
    answer_tokens = {
        'input_ids': [],
        'attention_mask': []
    }

    # Tokenize texts + questions
    for text, question in zip(texts, questions):
        encoding = tokenizer.encode(text + " " + question)
        input_ids = encoding.ids[:max_length]
        attention_mask = [1] * len(input_ids)
        input_ids += [0] * (max_length - len(input_ids))
        attention_mask += [0] * (max_length - len(attention_mask))
        data_tokens['input_ids'].append(input_ids)
        data_tokens['attention_mask'].append(attention_mask)

    # Tokenize answers
    for answer in answers:
        encoding = tokenizer.encode(answer)
        input_ids = encoding.ids[:max_length]
        attention_mask = [1] * len(input_ids)
        input_ids += [0] * (max_length - len(input_ids))
        attention_mask += [0] * (max_length - len(attention_mask))
        answer_tokens['input_ids'].append(input_ids)
        answer_tokens['attention_mask'].append(attention_mask)

    # Convert lists to tensors and wrap in BatchEncoding
    data_tokens = BatchEncoding({k: torch.tensor(v) for k, v in data_tokens.items()})
    answer_tokens = BatchEncoding({k: torch.tensor(v) for k, v in answer_tokens.items()})

    return [data_tokens, answer_tokens, qtypes]



def tokenize(df,max_length=512,tokenizer=None):
    texts       = df['texts'].values.tolist()
    questions   = df['questions'].values.tolist()
    answers     = df['answers'].values.tolist()
    qtypes      = df['qtypes'].values.tolist()


    data_tokens = tokenizer(texts, questions, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    answer_tokens = tokenizer(answers, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    return [data_tokens, answer_tokens, qtypes]

def load_matscholar(explanation = 0, res_dict = None):
    path = 'dataset/matscholar.json'.format(dir_path)
    df = pd.read_json(path, orient='split')
    text_list = []
    question_list = []
    answer_list = []
    qtype_list = []
    tokens = df['tokens'].tolist()
    labels = df['labels'].tolist()
    explanation_type = res_dict['explanation_type']
    exp_sent = ''
    exp_token = ''
    exp_label = ''
    if (explanation_type=='example'):
        idx = random.randint(0,len(labels))
        exp_sent = tokens[idx]
        ann = labels[idx]
        for k,v in ann.items():
            if (v!="O"):
                exp_token = k
                exp_label = v
            break
    for text,ann in zip(tokens, labels):
        for k,v in ann.items():
            if (v!="O"):
                text_list.append(text)
                if (explanation==0):
                    question_list.append('named entity recognition \t {}'.format(k))
                else:
                    if (explanation_type=='choice'):
                        candidates = res_dict['t_type_set']
                        exp_text = 'choices : ' + ','.join(candidates)
                        question_list.append('named entity recognition \t {}. {}'.format(k, exp_text))
                    elif (explanation_type=='schema'):
                        question_list.append('named entity recognition \t {}. schema: <entity type>'.format(k))
                    elif (explanation_type=='example'):
                        question_list.append('task: named entity recognition \t {}. example: {} named entity recognition \t {} answer: {}'.format(k, exp_sent, exp_token, exp_label))
                    else:
                        raise ValueError('invalid explanation type')
                answer_list.append(v)
                qtype_list.append(question_type_dict['ner'])
    return (text_list, question_list, answer_list, qtype_list)

def load_sofc_token(explanation = 0, res_dict = None):
    path = 'dataset/sofc_token.json'.format(dir_path)
    df = pd.read_json(path, orient='split')
    text_list = []
    question_list = []
    answer_list = []
    qtype_list = []
    tokens = df['tokens'].tolist()
    token_labels = df['token_labels'].tolist()
    slot_labels = df['slot_labels'].tolist()
    explanation_type = res_dict['explanation_type']
    exp_sent = ''
    exp_token = ''
    exp_slot = ''
    exp_token_label = ''
    exp_slot_label = ''
    if (explanation_type=='example'):
        idx = random.randint(0,len(slot_labels))
        exp_sent = ' '.join(tokens[idx])
        for t,l,s in zip(tokens[idx],token_labels[idx],slot_labels[idx]):
            if (l=="O" or s=="O"):
                continue
            exp_token = t
            exp_token_label = l
            exp_slot = t
            exp_slot_label = s
            break
    for token,label,slot in zip(tokens,token_labels,slot_labels):
        text = ' '.join(token)
        for t,l,s in zip(token,label,slot):
            if (l!="O"):
                text_list.append(text)
                if (explanation==0):
                    question_list.append('named entity recognition \t {}'.format(t))
                else:
                    if (explanation_type=='choice'):
                        candidates = res_dict['t_type_set']
                        exp_text = 'choices : ' + ','.join(candidates)
                        question_list.append('named entity recognition \t {}. {}'.format(t, exp_text))
                    elif (explanation_type=='schema'):
                        question_list.append('named entity recognition \t {}. schema: <entity type>'.format(t))
                    elif (explanation_type=='example'):
                        question_list.append('task: named entity recognition \t {}. example: {} named entity recognition \t {} answer: {}'.format(t, exp_sent, exp_token, exp_token_label))
                    else:
                        raise ValueError('invalid explanation type')
                answer_list.append(l)
                qtype_list.append(question_type_dict['ner'])
            if (s!="O"):
                text_list.append(text)
                if (explanation==0):
                    question_list.append('slot filling \t {}'.format(t))
                else:
                    if (explanation_type=='choice'):
                        candidates = res_dict['sf_type_set']
                        exp_text = 'choices : ' + ','.join(candidates)
                        question_list.append('slot filling \t {}. {}'.format(t, exp_text))
                    elif (explanation_type=='schema'):
                        question_list.append('slot filling \t {}. schema: <slot text>'.format(t))
                    elif (explanation_type=='example'):
                        question_list.append('task: slot filling \t {}. example: {} slot filling: {} answer: {}'.format(t, exp_sent, exp_slot, exp_slot_label))
                    else:
                        raise ValueError('invalid explanation type')
                answer_list.append(s)
                qtype_list.append(question_type_dict['sf'])
    return (text_list, question_list, answer_list, qtype_list)

def load_synthesis_procedures(explanation = 0, res_dict = None):
    path = 'dataset/synthesis_procedures.json'.format(dir_path)
    f = open(path)
    data_dict = json.load(f)
    items = data_dict['data']
    text_list = []
    question_list = []
    answer_list = []
    qtype_list = []
    explanation_type = res_dict['explanation_type']
    exp_text = ''
    exp_entity_name = ''
    exp_entity_type = ''
    exp_args = []
    exp_relation_type = ''
    exp_event_struc = []
    exp_trigger = ''
    if (explanation_type=='example'):
        idx = random.randint(0,len(items))
        item = items[idx]
        exp_text = item['text']
        for key,value in item['t_span_dict'].items():
            l,r = value[0],value[1]
            exp_entity_name = exp_text[l:r]
            exp_entity_type = item['t_type_dict'][key]
            break
        for key,value in item['r_args_dict'].items():
            exp_args = []
            for e in value:
                if e[0]=='T':
                    l,r = item['t_span_dict'][e]
                else:
                    l,r = item['t_span_dict'][item['e_trig_dict'][e]]
                exp_args.append(exp_text[l:r].replace(',',''))
            exp_relation_type = item['r_type_dict'][key]
            break
        for key,value in item['e_trig_dict'].items():
            exp_event_struc = []
            l,r = item['t_span_dict'][value]
            exp_trigger = exp_text[l:r]
            for arg in item['e_args_dict'][key]:
                l,r = item['t_span_dict'][arg[1]]
                exp_event_struc.append(str(exp_text[l:r]).replace(',','').replace(':','') + ":" + str(arg[0]).replace(',','').replace(':',''))
            if (len(exp_event_struc)==0):
                exp_event_struc.append('none:none')
            break
    for item in items:
        text = item['text']
        for key,value in item['t_span_dict'].items():
            l,r = value[0],value[1]
            entity_name = text[l:r]
            entity_type = item['t_type_dict'][key]
            text_list.append(text)
            if (explanation==0):
                question_list.append('named entity recognition:{}'.format(entity_name))
            else:
                if (explanation_type=='choice'):
                    candidates = res_dict['t_type_set']
                    exp_text = 'choices : ' + ','.join(candidates)
                    question_list.append('named entity recognition \t {}. {}'.format(entity_name, exp_text))
                elif (explanation_type=='schema'):
                    question_list.append('named entity recognition \t {}. schema: <entity type>'.format(entity_name))
                elif (explanation_type=='example'):
                    question_list.append('task: named entity recognition \t {}. example: {} named entity recognition \t {} answer: {}'.format(entity_name, exp_text, exp_entity_name, exp_entity_type))
                else:
                    raise ValueError('invalid explanation type')
            answer_list.append(entity_type)
            qtype_list.append(question_type_dict['ner'])
        for key,value in item['r_args_dict'].items():
            args = []
            for e in value:
                if e[0]=='T':
                    l,r = item['t_span_dict'][e]
                else:
                    l,r = item['t_span_dict'][item['e_trig_dict'][e]]
                args.append(text[l:r].replace(',',''))
            relation_type = item['r_type_dict'][key]
            text_list.append(text)
            if (explanation==0):
                question_list.append('relation classification' + '\t' + ','.join(args))
            else:
                if (explanation_type=='choice'):
                    candidates = res_dict['r_type_set']
                    exp_text = ' choices : ' + ','.join(candidates)
                    question_list.append('relation classification' + '\t' + ','.join(args) + exp_text)
                elif (explanation_type=='schema'):
                    question_list.append('relation classification' + '\t' + ','.join(args) + ' schema: <relation type>')
                elif (explanation_type=='example'):
                    question_list.append('task: relation classification' + '\t' + ','.join(args) + ' example: {}. relation classification \t {},{}. answer: {}'.format(exp_text, exp_args[0], exp_args[1], exp_relation_type))
                else:
                    raise ValueError('invalid explanation type')
            answer_list.append(relation_type)
            qtype_list.append(question_type_dict['rc'])
        for key,value in item['e_trig_dict'].items():
            event_struc = []
            l,r = item['t_span_dict'][value]
            trigger = text[l:r]
            for arg in item['e_args_dict'][key]:
                l,r = item['t_span_dict'][arg[1]]
                event_struc.append(str(text[l:r]).replace(',','').replace(':','') + ":" + str(arg[0]).replace(',','').replace(':',''))
            text_list.append(text)
            if (explanation==0):
                question_list.append('event extraction' + '\t' + trigger)
            else:
                if (explanation_type=='choice'):
                    candidates = res_dict['e_role_set']
                    exp_text = ' choices : ' + ','.join(candidates)
                    question_list.append('event extraction' + '\t' + trigger + exp_text)
                elif (explanation_type=='schema'):
                    schema = '. schema: argument mask:role mask'
                    question_list.append('event extraction' + '\t' + trigger + schema)
                elif (explanation_type=='example'):
                    question_list.append('task: event extraction \t {}. example: {}. event extraction \t {} answer: {}'.format(trigger, exp_text, exp_trigger, ','.join(exp_event_struc)))
                else:
                    raise ValueError('invalid explanation type')
            if (len(event_struc)==0):
                event_struc.append('none:none')
            answer_list.append(','.join(event_struc))
            qtype_list.append(question_type_dict['ee'])
    return (text_list, question_list, answer_list, qtype_list)

def load_sc_comics(explanation = 0, res_dict = None):
    path = 'dataset/sc_comics.json'.format(dir_path)
    f = open(path)
    data_dict = json.load(f)
    items = data_dict['data']
    text_list = []
    question_list = []
    answer_list = []
    qtype_list = []
    explanation_type = res_dict['explanation_type']
    exp_text = ''
    exp_entity_name = ''
    exp_entity_type = ''
    exp_args = []
    exp_relation_type = ''
    exp_event_struc = []
    exp_trigger = ''
    if (explanation_type=='example'):
        idx = random.randint(0,len(items))
        item = items[idx]
        exp_text = item['text']
        for key,value in item['t_span_dict'].items():
            l,r = value[0],value[1]
            exp_entity_name = exp_text[l:r]
            exp_entity_type = item['t_type_dict'][key]
            break
        for key,value in item['r_args_dict'].items():
            exp_args = []
            for e in value:
                if e[0]=='T':
                    l,r = item['t_span_dict'][e]
                else:
                    l,r = item['t_span_dict'][item['e_trig_dict'][e]]
                exp_args.append(exp_text[l:r].replace(',',''))
            exp_relation_type = item['r_type_dict'][key]
            break
        for key,value in item['e_trig_dict'].items():
            exp_event_struc = []
            l,r = item['t_span_dict'][value]
            exp_trigger = exp_text[l:r]
            for arg in item['e_args_dict'][key]:
                l,r = item['t_span_dict'][arg[1]]
                exp_event_struc.append(str(exp_text[l:r]).replace(',','').replace(':','') + ":" + str(arg[0]).replace(',','').replace(':',''))
            if (len(exp_event_struc)==0):
                exp_event_struc.append('none:none')
            break
    for item in items:
        text = item['text']
        for key,value in item['t_span_dict'].items():
            l,r = value[0],value[1]
            entity_name = text[l:r]
            entity_type = item['t_type_dict'][key]
            text_list.append(text)
            if (explanation==0):
                question_list.append('named entity recognition:{}'.format(entity_name))
            else:
                if (explanation_type=='choice'):
                    candidates = res_dict['t_type_set']
                    exp_text = 'choices : ' + ','.join(candidates)
                    question_list.append('named entity recognition \t {}. {}'.format(entity_name, exp_text))
                elif (explanation_type=='schema'):
                    question_list.append('named entity recognition \t {}. schema: <entity type>'.format(entity_name))
                elif (explanation_type=='example'):
                    question_list.append('task: named entity recognition \t {}. example: {} named entity recognition \t {} answer: {}'.format(entity_name, exp_text, exp_entity_name, exp_entity_type))
                else:
                    raise ValueError('invalid explanation type')
            answer_list.append(entity_type)
            qtype_list.append(question_type_dict['ner'])
        for key,value in item['r_args_dict'].items():
            args = []
            for e in value:
                if e[0]=='T':
                    l,r = item['t_span_dict'][e]
                else:
                    l,r = item['t_span_dict'][item['e_trig_dict'][e]]
                args.append(text[l:r].replace(',',''))
            relation_type = item['r_type_dict'][key]
            text_list.append(text)
            if (explanation==0):
                question_list.append('relation classification' + '\t' + ','.join(args))
            else:
                if (explanation_type=='choice'):
                    candidates = res_dict['r_type_set']
                    exp_text = ' choices : ' + ','.join(candidates)
                    question_list.append('relation classification' + '\t' + ','.join(args) + exp_text)
                elif (explanation_type=='schema'):
                    question_list.append('relation classification' + '\t' + ','.join(args) + ' schema: <relation type>')
                elif (explanation_type=='example'):
                    question_list.append('task: relation classification' + '\t' + ','.join(args) + ' example: {}. relation classification \t {},{}. answer: {}'.format(exp_text, exp_args[0], exp_args[1], exp_relation_type))
                else:
                    raise ValueError('invalid explanation type')
            answer_list.append(relation_type)
            qtype_list.append(question_type_dict['rc'])
        for key,value in item['e_trig_dict'].items():
            event_struc = []
            l,r = item['t_span_dict'][value]
            trigger = text[l:r]
            for arg in item['e_args_dict'][key]:
                l,r = item['t_span_dict'][arg[1]]
                event_struc.append(str(text[l:r]).replace(',','').replace(':','') + ":" + str(arg[0]).replace(',','').replace(':',''))
            text_list.append(text)
            if (explanation==0):
                question_list.append('event extraction' + '\t' + trigger)
            else:
                if (explanation_type=='choice'):
                    candidates = res_dict['e_role_set']
                    exp_text = ' choices : ' + ','.join(candidates)
                    question_list.append('event extraction' + '\t' + trigger + exp_text)
                elif (explanation_type=='schema'):
                    schema = '. schema: argument mask:role mask'
                    question_list.append('event extraction' + '\t' + trigger + schema)
                elif (explanation_type=='example'):
                    question_list.append('task: event extraction \t {}. example: {}. event extraction \t {} answer: {}'.format(trigger, exp_text, exp_trigger, ','.join(exp_event_struc)))
                else:
                    raise ValueError('invalid explanation type')
            if (len(event_struc)==0):
                event_struc.append('none:none')
            answer_list.append(','.join(event_struc))
            qtype_list.append(question_type_dict['ee'])
    return (text_list, question_list, answer_list, qtype_list)

def load_glass(explanation = 0, res_dict = None):
    path = 'dataset/glass_non_glass.json'.format(dir_path)
    df = pd.read_json(path, orient='split')
    text_list = []
    question_list = []
    answer_list = []
    qtype_list = []
    abstracts = df['Abstract'].tolist()
    labels = df['Label'].tolist()
    explanation_type = res_dict['explanation_type']
    exp_sent = ''
    exp_label = ''
    if (explanation_type=='example'):
        idx = random.randint(0,len(labels))
        exp_sent = abstracts[idx]
        exp_label = labels[idx]
    for text,label in zip(abstracts,labels):
        text_list.append(text)
        if (explanation==0):
            question_list.append('paragraph classification')
        else:
            if (explanation_type=='choice'):
                question_list.append('paragraph classification. choices : yes,no')
            elif (explanation_type=='schema'):
                question_list.append('paragraph classification. schema : <yes> or <no>')
            elif (explanation_type=='example'):
                question_list.append('task: paragraph classification. example : task: {} answer: {}'.format(exp_sent, exp_label))
            else:
                assert ValueError('invalid explanation type')
        answer = 'yes' if int(label)==1 else 'no'
        answer_list.append(answer)
        qtype_list.append(question_type_dict['pc'])
    return (text_list, question_list, answer_list, qtype_list)

def load_re(explanation = 0, res_dict = None):
    path = 'dataset/structured_re.json'.format(dir_path)
    data = open(path).read().strip().split('\n')
    text_list = []
    question_list = []
    answer_list = []
    qtype_list = []
    explanation_type = res_dict['explanation_type']
    exp_sent = ''
    exp_args = []
    exp_answer = ''
    if (explanation_type=='example'):
        idx = random.randint(0,len(data))
        line = data[idx]
        j = json.loads(line)
        exp_sent = j['sentText']
        for rel in j['relationMentions']:
            exp_args = [rel['arg1Text'],rel['arg2Text']]
            exp_answer = rel['relText']
    for line in data:
        j = json.loads(line)
        text = j['sentText']
        for rel in j['relationMentions']:
            args = [rel['arg1Text'],rel['arg2Text']]
            answer = rel['relText']
            text_list.append(text)
            if (explanation==0):
                question_list.append('relation classification' + '\t' + ','.join(args))
            else:
                if (explanation_type=='choice'):
                    candidates = res_dict['r_type_set']
                    exp_text = ' choices : ' + ','.join(candidates)
                    question_list.append('relation classification' + '\t' + ','.join(args) + exp_text)
                elif (explanation_type=='schema'):
                    question_list.append('relation classification' + '\t' + ','.join(args) + ' schema: <relation type>')
                elif (explanation_type=='example'):
                    question_list.append('task: relation classification' + '\t' + ','.join(args) + ' example: {}. relation classification \t {},{}. answer: {}'.format(exp_sent, exp_args[0], exp_args[1], exp_answer))
                else:
                    raise ValueError('invalid explanation type')
            answer_list.append(answer)
            qtype_list.append(question_type_dict['rc'])
    return (text_list, question_list, answer_list, qtype_list)

def load_synthesis_actions(explanation = 0, res_dict = None):
    path = 'dataset/synthesis_actions.json'.format(dir_path)
    data = json.load(open(path))
    text_list = []
    question_list = []
    answer_list = []
    qtype_list = []
    exp_sent = ''
    exp_token = ''
    exp_answer = ''
    explanation_type = res_dict['explanation_type']
    if (explanation_type=='example'):
        idx = random.randint(0,len(data))
        j = data[idx]
        exp_sent = j['sentence']
        for ann in j['annotations']:
            token = ann['token']
            tag = ann['tag']
            if (len(tag)==0):
                continue
            exp_token = token
            exp_answer = tag
            break
    for j in data:
        text = j['sentence']
        for ann in j['annotations']:
            token = ann['token']
            tag = ann['tag']
            if (len(tag)==0):
                continue
            text_list.append(text)
            if (explanation==0):
                question_list.append('synthesis action retrieval \t {}'.format(token))
            else:
                if (explanation_type=='choice'):
                    candidates = res_dict['sar_set']
                    exp_text = ' choices : ' + ','.join(candidates)
                    question_list.append('synthesis action retrieval \t {}. {}'.format(token, exp_text))
                elif (explanation_type=='schema'):
                    question_list.append('synthesis action retrieval \t {}. schema : <action>'.format(token))
                elif (explanation_type=='example'):
                    question_list.append('task: synthesis action retrieval \t {}. example : {}. synthesis action retrieval \t {}. answer: {}'.format(token, exp_sent, exp_token, exp_answer))
                else:
                    raise ValueError('invalid explanation type')
            answer_list.append(tag)
            qtype_list.append(question_type_dict['sar'])
    return (text_list, question_list, answer_list, qtype_list)

def load_sofc_sent(explanation = 0, res_dict = None):
    path = 'dataset/sofc_sent.json'.format(dir_path)
    df = pd.read_json(path, orient='split')
    text_list = []
    question_list = []
    answer_list = []
    qtype_list = []
    sents = df['sents'].tolist()
    labels = df['sent_labels'].tolist()
    explanation_type = res_dict['explanation_type']
    exp_sent = ''
    exp_label = ''
    if (explanation_type=='example'):
        idx = random.randint(0,len(labels))
        exp_sent = sents[idx]
        exp_label = labels[idx]
    for sent,label in zip(sents,labels):
        text_list.append(sent)
        if (explanation==0):
            question_list.append('sentence classification')
        else:
            if (explanation_type=='choice'):
                question_list.append('sentence classification. choices : yes,no')
            elif (explanation_type=='schema'):
                question_list.append('sentence classification. schema : <yes> or <no>')
            elif (explanation_type=='example'):
                question_list.append('task: sentence classification. example : task: {} answer: {}'.format(exp_sent, exp_label))
            else:
                assert ValueError('invalid explanation type')
        answer = 'yes' if int(label)==1 else 'no'
        answer_list.append(answer)
        qtype_list.append(question_type_dict['sc'])
    return (text_list, question_list, answer_list, qtype_list)

def format_data(datasets, tasks, explanation = 0, res_dict = None, lower=True):
    processed_data = dict()
    processed_data['texts'] = []
    processed_data['questions'] = []
    processed_data['answers'] = []
    processed_data['qtypes'] = []
    for dataset in datasets:
        if dataset=='matscholar.json':
            text_list, question_list, answer_list, qtype_list = load_matscholar(explanation,res_dict)
            processed_data['texts'] += text_list
            processed_data['questions'] += question_list
            processed_data['answers'] += answer_list
            processed_data['qtypes'] += qtype_list
            print('text len = {} question len = {} answer len = {} qtype len = {}'.format(len(text_list),len(question_list),len(answer_list),len(qtype_list)))
            print('dataset = {} size = {}'.format(dataset,len(processed_data['texts'])))
        if dataset=='sofc_token.json':
            text_list, question_list, answer_list, qtype_list = load_sofc_token(explanation,res_dict)
            processed_data['texts'] += text_list
            processed_data['questions'] += question_list
            processed_data['answers'] += answer_list
            processed_data['qtypes'] += qtype_list
            print('text len = {} question len = {} answer len = {} qtype len = {}'.format(len(text_list),len(question_list),len(answer_list),len(qtype_list)))
            print('dataset = {} size = {}'.format(dataset,len(processed_data['texts'])))
        if dataset=='synthesis_procedures.json':
            text_list, question_list, answer_list, qtype_list = load_synthesis_procedures(explanation,res_dict)
            processed_data['texts'] += text_list
            processed_data['questions'] += question_list
            processed_data['answers'] += answer_list
            processed_data['qtypes'] += qtype_list
            print('text len = {} question len = {} answer len = {} qtype len = {}'.format(len(text_list),len(question_list),len(answer_list),len(qtype_list)))
            print('dataset = {} size = {}'.format(dataset,len(processed_data['texts'])))
        if dataset=='sc_comics.json':
            text_list, question_list, answer_list, qtype_list = load_sc_comics(explanation,res_dict)
            processed_data['texts'] += text_list
            processed_data['questions'] += question_list
            processed_data['answers'] += answer_list
            processed_data['qtypes'] += qtype_list
            print('text len = {} question len = {} answer len = {} qtype len = {}'.format(len(text_list),len(question_list),len(answer_list),len(qtype_list)))
            print('dataset = {} size = {}'.format(dataset,len(processed_data['texts'])))
        if dataset=='glass_non_glass.json':
            text_list, question_list, answer_list, qtype_list = load_glass(explanation,res_dict)
            processed_data['texts'] += text_list
            processed_data['questions'] += question_list
            processed_data['answers'] += answer_list
            processed_data['qtypes'] += qtype_list
            print('text len = {} question len = {} answer len = {} qtype len = {}'.format(len(text_list),len(question_list),len(answer_list),len(qtype_list)))
            print('dataset = {} size = {}'.format(dataset,len(processed_data['texts'])))
        if dataset=='structured_re.json':
            text_list, question_list, answer_list, qtype_list = load_re(explanation,res_dict)
            processed_data['texts'] += text_list
            processed_data['questions'] += question_list
            processed_data['answers'] += answer_list
            processed_data['qtypes'] += qtype_list
            print('text len = {} question len = {} answer len = {} qtype len = {}'.format(len(text_list),len(question_list),len(answer_list),len(qtype_list)))
            print('dataset = {} size = {}'.format(dataset,len(processed_data['texts'])))
        if dataset=='synthesis_actions.json':
            text_list, question_list, answer_list, qtype_list = load_synthesis_actions(explanation,res_dict)
            processed_data['texts'] += text_list
            processed_data['questions'] += question_list
            processed_data['answers'] += answer_list
            processed_data['qtypes'] += qtype_list
            print('text len = {} question len = {} answer len = {} qtype len = {}'.format(len(text_list),len(question_list),len(answer_list),len(qtype_list)))
            print('dataset = {} size = {}'.format(dataset,len(processed_data['texts'])))
        if dataset=='sofc_sent.json':
            text_list, question_list, answer_list, qtype_list = load_sofc_sent(explanation,res_dict)
            processed_data['texts'] += text_list
            processed_data['questions'] += question_list
            processed_data['answers'] += answer_list
            processed_data['qtypes'] += qtype_list
            print('text len = {} question len = {} answer len = {} qtype len = {}'.format(len(text_list),len(question_list),len(answer_list),len(qtype_list)))
            print('dataset = {} size = {}'.format(dataset,len(processed_data['texts'])))
    df = pd.DataFrame(processed_data)
    df = df.loc[df['qtypes'].isin(tasks)]

    if lower:
        df['texts'] = df['texts'].apply(lambda x:x.lower())
        df['questions'] = df['questions'].apply(lambda x:x.lower())
        df['answers'] = df['answers'].apply(lambda x:x.lower())
    return df

def split_data(df, setting = 'ood', train_size = 0.05, even_split = False):
    train_df, test_df = None, None
    if (setting == 'ood'):
        train_df_list = []
        test_df_list  = []
        for qtype in df['qtypes'].unique():
            tmp_df = df[df.qtypes==qtype]
            print('qtype = {} datasize = {}'.format(qtype, len(tmp_df)))
            if (qtype in [0,2,4,6]):
                unique_answers = list(tmp_df['answers'].unique())
                train_answers, test_answers = train_test_split(unique_answers, train_size=train_size)
                train_df_list.append(tmp_df.loc[tmp_df['answers'].isin(train_answers)])
                test_df_list.append(tmp_df.loc[tmp_df['answers'].isin(test_answers)])
            elif (qtype==3):
                answers = tmp_df['answers'].tolist()
                unique_types = set()
                for answer in answers:
                    for args in answer.split(','):
                        unique_types.add(args.split(':')[1])
                train_answers, test_answers = train_test_split(list(unique_types), train_size=train_size)
                def isin_train(x):
                    isin = False
                    for args in x.split(','):
                        role_type = args.split(':')[1]
                        if (role_type in train_answers):
                            return True
                    return isin
                def isin_test(x):
                    isin = False
                    for args in x.split(','):
                        role_type = args.split(':')[1]
                        if (role_type in test_answers):
                            return True
                    return isin
                train_df_list.append(tmp_df.loc[tmp_df['answers'].apply(lambda x:isin_train(x))])
                test_df_list.append(tmp_df.loc[tmp_df['answers'].apply(lambda x:isin_test(x))])
            else:
                tmp_train_df, tmp_test_df = train_test_split(tmp_df, train_size=train_size)
                train_df_list.append(tmp_train_df)
                test_df_list.append(tmp_test_df)
        train_df = pd.concat(train_df_list,ignore_index=True)
        test_df  = pd.concat(test_df_list,ignore_index=True)
    elif (setting == 'low_resource'):
        if (even_split is False):
            train_df_list = []
            test_df_list  = []
            for qtype in df['qtypes'].unique():
                tmp_df = df[df.qtypes==qtype]
                tmp_train_df, tmp_test_df = train_test_split(tmp_df, train_size=train_size)
                train_df_list.append(tmp_train_df)
                test_df_list.append(tmp_test_df)
                print('qtype = {} datasize = {}'.format(qtype, len(tmp_df)))
            train_df = pd.concat(train_df_list,ignore_index=True)
            test_df  = pd.concat(test_df_list,ignore_index=True)
        else:
            train_df_list = []
            test_df_list = []
            for qtype in df['qtypes'].unique():
                tmp_df = df[df.qtypes==qtype]
                if (qtype in [0,1,2,4,5,6]):
                    unique_answers = list(tmp_df['answers'].unique())
                    for answer in unique_answers:
                        tmp_df_2 = tmp_df.loc[tmp_df['answers']==answer]
                        if len(tmp_df_2)*train_size>1:
                            tmp_train_df, tmp_test_df = train_test_split(tmp_df_2, train_size=train_size)
                            train_df_list.append(tmp_train_df)
                            test_df_list.append(tmp_test_df)
                        else:
                            tmp_train_df, tmp_test_df = train_test_split(tmp_df_2, train_size=default_sample_num)
                            train_df_list.append(tmp_train_df)
                            test_df_list.append(tmp_test_df)
                else:
                    answers = tmp_df['answers'].tolist()
                    unique_types = set()
                    for answer in answers:
                        for args in answer.split(','):
                            unique_types.add(args.split(':')[1])
                    def is_type(x,unique_type):
                        isin = False
                        for args in x.split(','):
                            role_type = args.split(':')[1]
                            if (role_type==unique_type):
                                return True
                        return isin
                    for unique_type in unique_types:
                        tmp_df_2 = tmp_df.loc[tmp_df['answers'].apply(lambda x:is_type(x,unique_type))]
                        if len(tmp_df_2)*train_size>1:
                            tmp_train_df, tmp_test_df = train_test_split(tmp_df_2, train_size=train_size)
                            train_df_list.append(tmp_train_df)
                            test_df_list.append(tmp_test_df)
                        else:
                            tmp_train_df, tmp_test_df = train_test_split(tmp_df_2, train_size=default_sample_num)
                            train_df_list.append(tmp_train_df)
                            test_df_list.append(tmp_test_df)
            train_df = pd.concat(train_df_list,ignore_index=True)
            test_df  = pd.concat(test_df_list,ignore_index=True)
    else:
        raise ValueError('settings invalid!!!')

    return train_df, test_df

def get_res_dict(df):
    res_dict = dict()
    for qtype in df['qtypes'].unique():
        tmp_df = df[df.qtypes==qtype]
        if (qtype==0):
            answer_set = [x.lower().strip().replace(' ', '') for x in tmp_df['answers'].unique()]
            answer_map = dict(zip(answer_set,list(range(len(answer_set)))))
            res_dict['t_type_set'] = answer_set
            res_dict['t_type_dict'] = answer_map
        if (qtype==1):
            res_dict['pc_type_dict'] = {'yes':1,'no':0}
            res_dict['pc_type_set'] = ['yes','no']
        if (qtype==2):
            answer_set = [x.lower().strip().replace(' ', '') for x in tmp_df['answers'].unique()]
            answer_map = dict(zip(answer_set,list(range(len(answer_set)))))
            res_dict['r_type_set'] = answer_set
            res_dict['r_type_dict'] = answer_map
        if (qtype==3):
            answers = tmp_df['answers'].tolist()
            unique_types = set()
            for answer in answers:
                for args in answer.split(','):
                    unique_types.add(args.split(':')[1])
            answer_set = [x.lower().strip().replace(' ', '') for x in unique_types]
            answer_map = dict(zip(answer_set,list(range(len(answer_set)))))
            res_dict['e_role_set'] = answer_set
            res_dict['e_role_dict'] = answer_map
        if (qtype==4):
            answer_set = [x.lower().strip().replace(' ', '') for x in tmp_df['answers'].unique()]
            answer_map = dict(zip(answer_set,list(range(len(answer_set)))))
            res_dict['sar_set'] = answer_set
            res_dict['sar_dict'] = answer_map
        if (qtype==5):
            res_dict['sc_type_dict'] = {'yes':1,'no':0}
            res_dict['sc_type_set'] = ['yes','no']
        if (qtype==6):
            answer_set = [x.lower().strip().replace(' ', '') for x in tmp_df['answers'].unique()]
            answer_map = dict(zip(answer_set,list(range(len(answer_set)))))
            res_dict['sf_type_set'] = answer_set
            res_dict['sf_type_dict'] = answer_map
    return res_dict

def build_dataloader(max_length = 512, batch_size = 4, tasks = [0,1,2,3,4,5,6], explanation = 0, setting = 'ood', train_size = 0.25, base_model = 'matscibert', explanation_type = 'choice', even_split = False, hf_model_name: str = 'allenai/scibert_scivocab_uncased'):
    datasets = ['matscholar.json','sofc_token.json','synthesis_procedures.json','sc_comics.json','glass_non_glass.json','structured_re.json','synthesis_actions.json','sofc_sent.json']
    lower = True
    if (base_model=='BPE'):
        tokenizer_path = "FIX"
        tokenizer = ByteLevelBPETokenizer.from_file(
            vocab_filename=os.path.join(tokenizer_path, "vocab.json"),
            merges_filename=os.path.join(tokenizer_path, "merges.txt")
        )
    elif (base_model=='WordPiece'):
        tokenizer = AutoTokenizer.from_pretrained("{CHECKPOINT_PATH}")
    elif (base_model=='PickyBPE'):
        tokenizer_path = "FIX"
        tokenizer = ByteLevelBPETokenizer.from_file(
            vocab_filename=os.path.join(tokenizer_path, "vocab.json"),
            merges_filename=os.path.join(tokenizer_path, "merges.txt")
        )
    elif (base_model=='SAGE'):
        tokenizer_path = "FIX"
        tokenizer = ByteLevelBPETokenizer.from_file(
            vocab_filename=os.path.join(tokenizer_path, "vocab.json"),
            merges_filename=os.path.join(tokenizer_path, "merges.txt")
        )

    else:
        raise ValueError('basemodel invalid!!!')
    
    
    if isinstance(matscibert_tokenizer, ByteLevelBPETokenizer):

        # Instead of using convert_tokens_to_ids, use the tokenizer's vocab
        vocab_dict = tokenizer.get_vocab()  # Get the vocabulary as a dictionary
        pad_idx = vocab_dict.get('[PAD]', None)  # Get the index for the [PAD] token, if it exists
        vocab_size = len(vocab_dict)  # Get the size of the vocabulary
    else:
        pad_idx = tokenizer.convert_tokens_to_ids('[PAD]')
        vocab_size = tokenizer.vocab_size
    res_dict = {'explanation_type':explanation_type}
    tmp_df = format_data(datasets, tasks, 0, res_dict, lower)
    res_dict = get_res_dict(tmp_df)
    res_dict['tokenizer']  = tokenizer
    res_dict['explanation_type'] = explanation_type
    df = format_data(datasets, tasks, explanation, res_dict, lower)
    train_df, test_df = split_data(df,setting,train_size,even_split)
    print('train_size = {} test_size = {}'.format(len(train_df),len(test_df)))
    
    
    if isinstance(matscibert_tokenizer, ByteLevelBPETokenizer):
        train_data = tokenize_bpe(train_df, max_length, tokenizer)
        test_data  = tokenize_bpe(test_df, max_length, tokenizer)
    else:
        train_data = tokenize(train_df, max_length, tokenizer)
        test_data  = tokenize(test_df, max_length, tokenizer)
    train_dataset = CustomDataset(train_data)
    test_dataset  = CustomDataset(test_data)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return (train_loader, test_loader, pad_idx, vocab_size, res_dict)
