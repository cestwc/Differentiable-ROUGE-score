from torch import nn
from transformers import Trainer

from losses import ROUGELoss

class RougeTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs['labels'].cuda()
		outputs = model(input_ids = inputs['input_ids'].cuda(), attention_mask = inputs['attention_mask'].cuda())
		logits = outputs.logits
		loss_fct = ROUGELoss().cuda()
		loss = loss_fct(logits, labels)
		return (loss, outputs) if return_outputs else loss
