import os
import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from datasets import load_dataset

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--preds_dir', required=True, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--model_name', default='m3rg-iitd/matscibert', type=str)
    parser.add_argument('--seeds', nargs='+', default=None, type=int)
    parser.add_argument('--lm_lrs', nargs='+', default=None, type=float)
    parser.add_argument('--non_lm_lr', default=3e-4, type=float)
    return parser.parse_args()


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    acc = (preds == p.label_ids).astype(np.float32).mean().item()
    return {'accuracy': acc}


def main():
    args = parse_args()

    cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
    output_dir = ensure_dir(args.output_dir)
    preds_dir = ensure_dir(args.preds_dir)

    if args.seeds is None:
        args.seeds = [42, 43, 44]
    if args.lm_lrs is None:
        args.lm_lrs = [2e-5, 3e-5, 5e-5]

    data_files = {split: os.path.join(args.dataset_dir, f'{split}.csv') for split in ['train', 'val', 'test']}
    datasets = load_dataset('csv', data_files=data_files, cache_dir=cache_dir)

    label_list = datasets['train'].unique('Label')
    num_labels = len(label_list)

    max_seq_length = 512
    config_kwargs = {
        'num_labels': num_labels,
        'cache_dir': cache_dir,
        'revision': 'main',
        'use_auth_token': None,
    }
    config = AutoConfig.from_pretrained(args.model_name, **config_kwargs)

    tokenizer_kwargs = {
        'cache_dir': cache_dir,
        'use_fast': True,
        'revision': 'main',
        'use_auth_token': None,
        'model_max_length': max_seq_length
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)

    def preprocess_function(examples):
        result = tokenizer(examples['Abstract'], padding=False, max_length=max_seq_length, truncation=True)
        result['label'] = [l for l in examples['Label']]
        return result

    tokenized = datasets.map(preprocess_function, batched=True)
    train_dataset = tokenized['train']
    val_dataset = tokenized['val']
    test_dataset = tokenized['test']

    metric_for_best_model = 'accuracy'

    best_lr = None
    best_val = -1.0
    best_val_acc_list = None
    best_test_acc_list = None

    for lr in args.lm_lrs:
        val_acc, test_acc = [], []
        for SEED in args.seeds:
            set_seed(SEED)
            training_args = TrainingArguments(
                num_train_epochs=10,
                output_dir=output_dir,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                evaluation_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=True,
                save_total_limit=2,
                warmup_ratio=0.1,
                learning_rate=lr,
                seed=SEED,
                report_to=[],
                logging_steps=1000000,
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                from_tf=False,
                config=config,
                cache_dir=cache_dir,
                revision='main',
                use_auth_token=None,
            )

            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not 'bert' in n], 'lr': args.non_lm_lr},
                {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': lr}
            ]
            optimizer_kwargs = {'betas': (0.9, 0.999), 'eps': 1e-8}
            optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                optimizers=(optimizer, None),
            )

            _ = trainer.train()
            val_result = trainer.evaluate()
            test_result = trainer.evaluate(test_dataset)
            val_acc.append(val_result[f'eval_{metric_for_best_model}'])
            test_acc.append(test_result[f'eval_{metric_for_best_model}'])

        mean_val = float(np.mean(val_acc))
        if mean_val > best_val:
            best_val = mean_val
            best_lr = lr
            best_val_acc_list = val_acc
            best_test_acc_list = test_acc

    summary = {
        'best_lr': best_lr,
        'val_accuracy_list': best_val_acc_list,
        'test_accuracy_list': best_test_acc_list,
        'seeds': args.seeds,
        'model_name': args.model_name,
    }
    with open(os.path.join(preds_dir, 'pc_summary.json'), 'w') as f:
        json.dump(summary, f)

    print('[run_pc] best_lr:', best_lr)
    print('[run_pc] val acc list:', best_val_acc_list)
    print('[run_pc] test acc list:', best_test_acc_list)


if __name__ == '__main__':
    main()


