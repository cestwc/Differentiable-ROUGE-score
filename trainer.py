from torch import nn
from transformers import Trainer

class RougeTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs.get("labels")
		outputs = model(**inputs)
		logits = outputs.get('logits')
		loss_fct = ROUGELoss()
		loss = loss_fct(logits, labels).mean()
		return (loss, outputs) if return_outputs else loss
