from gramgen import GGTrainer

from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer
import torch

model_name = 'bert-base-uncased'

model = AutoModelForMaskedLM.from_pretrained(model_name)

from utils.datasets_config import get_dataset

dset = get_dataset('gigaword', model_name)
print(dset)


training_args = TrainingArguments(
    # evaluation_strategy = "steps",
    output_dir= model_name + '-maskedlm',
    overwrite_output_dir=True,
    num_train_epochs=1,
    max_steps = 700_000,
    per_device_train_batch_size=16,
    save_steps=5_000,
    save_total_limit=20,
    prediction_loss_only=True,
    dataloader_num_workers=4,
    # learning_rate=3e-4,
    # logging_steps = 5,
    # eval_steps = 5,
    # metric_for_best_model = 'f1',
    # load_best_model_at_end=True,
)

def collate(batch):
    batch = ({k: torch.nn.utils.rnn.pad_sequence([dic[k] for dic in batch], batch_first=True, padding_value=1) for k in batch[0]})
    # batch['input_ids'][batch['input_ids'] == -100] = 1
    batch['attention_mask'] = (batch['input_ids'] != 1).long()
    batch['labels'] = torch.nn.functional.pad(batch['labels'], (0, batch['input_ids'].shape[1] - batch['labels'].shape[1], 0, 0), 'constant', 1)
    return batch

trainer = GGTrainer(
    model = model,
    args = training_args,
    train_dataset = dset['train'].shuffle(1234),
    eval_dataset = dset['validation'].shard(300, 1),
    data_collator = collate,
    # compute_metrics = lambda x: print(x),
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=7)],
)

trainer.train(resume_from_checkpoint = False)

trainer.evaluate()
