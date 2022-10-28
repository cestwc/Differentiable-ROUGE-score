from trainer import RougeTrainer
from datasets import load_dataset, load_from_disk

cnn_dailymail = load_dataset('ccdv/cnn_dailymail', '3.0.0')

from transformers import BertTokenizerFast, BertForMaskedLM
import torch

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

def tokenize(e):
    article = tokenizer(e['article'], max_length=512, truncation=True, padding = 'max_length')
    article['labels'] = tokenizer(e['highlights'], max_length=128, truncation=True, padding = 'max_length')['input_ids']
    
    return article

cnn_dailymail_tokenized = cnn_dailymail.map(tokenize, batched=True)

cnn_dailymail_tokenized.set_format(type='torch', columns=['input_ids','labels', 'attention_mask'])

print(f"Number of training examples: {len(cnn_dailymail_tokenized['train'])}")
print(f"Number of validation examples: {len(cnn_dailymail_tokenized['validation'])}")
print(f"Number of testing examples: {len(cnn_dailymail_tokenized['test'])}")

import torch
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The summarizer has {count_parameters(model):,} trainable parameters')


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir= 'Bert',
    overwrite_output_dir=True,
    num_train_epochs=1,
    max_steps = 70_000,
    per_device_train_batch_size=8,
    save_steps=5_000,
    save_total_limit=20,
    prediction_loss_only=True,
    # learning_rate=3e-4,
    # logging_steps = 2
)

trainer = RougeTrainer(
    model = model,
    args = training_args,
    train_dataset = cnn_dailymail_tokenized['train'].shuffle(1234),
    eval_dataset = cnn_dailymail_tokenized['validation'],
    # data_collator = collate,
)

trainer.train(resume_from_checkpoint = False)

trainer.evaluate()