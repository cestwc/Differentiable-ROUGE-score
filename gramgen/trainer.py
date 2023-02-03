from transformers import Trainer
from .loss import GramGenerateLoss

class GGTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs['labels'].cuda()
		outputs = model(input_ids = inputs['input_ids'].cuda(), attention_mask = inputs['attention_mask'].cuda())
		logits = outputs.logits
		loss_fct = GramGenerateLoss(reduction = 'mean', ignore_index = 1).cuda()
		loss = loss_fct(logits.transpose(1, 2), labels)
		return (loss, outputs) if return_outputs else loss
