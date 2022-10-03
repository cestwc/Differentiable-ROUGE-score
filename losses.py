import torch
import torch.nn.functional as F

from conv_core import sudoku, diagonal2

class ROUGELoss(torch.nn.Module):
	def __init__(self, reduction = 'mean', n = 1, metrics='fmeasure'):
		super(ROUGELoss, self).__init__()
		self.reduction = reduction
		self.n = n
		self.metrics = metrics
	def forward(self, logits, labels):
		# overlap = F.embedding(labels.view(-1), F.softmax(logits.view(-1, logits.shape[-1]), dim=-1).T)
		overlap =  torch.stack([F.embedding(b, a.T) for a, b in zip(logits.softmax(-1), labels)]).unsqueeze(1)
		numerator = sudoku(overlap).sum((-2, -1))
		denominators = torch.tensor(overlap.shape[-2:]) - self.n + 1
		if self.metrics == 'fmeasure':
			return 2 * numerator / sum(denominators)
		elif self.metrics == 'precision':
			return numerator / denominators[0]
		elif self.metrics == 'recall':
			return numerator / denominators[1]
		else:
			return overlap
