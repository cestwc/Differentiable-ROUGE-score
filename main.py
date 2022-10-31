from trainer import RougeTrainer

from transformers import AutoModelForMaskedLM, TrainingArguments
import torch

model_name = 'bert-base-uncased'

model = AutoModelForMaskedLM.from_pretrained(model_name)

from utils.datasets_config import get_dataset

dset = get_dataset('gigaword', model_name)
print(dset)


training_args = TrainingArguments(
    output_dir= model_name + '-maskedlm',
    overwrite_output_dir=True,
    num_train_epochs=1,
    max_steps = 700_000,
    per_device_train_batch_size=4,
    save_steps=5_000,
    save_total_limit=20,
    prediction_loss_only=True,
    dataloader_num_workers=4,
    # learning_rate=3e-4,
    # logging_steps = 2
)

def collate(batch):
    batch = ({k: torch.nn.utils.rnn.pad_sequence([dic[k] for dic in batch], batch_first=True, padding_value=1) for k in batch[0]})
    # batch['input_ids'][batch['input_ids'] == -100] = 1
    batch['attention_mask'] = (batch['input_ids'] != 1).long()
    return batch

trainer = RougeTrainer(
    model = model,
    args = training_args,
    train_dataset = dset['train'].shuffle(1234).shard(30, 0),
    eval_dataset = dset['validation'],
    data_collator = collate,
)

trainer.train(resume_from_checkpoint = False)

trainer.evaluate()
