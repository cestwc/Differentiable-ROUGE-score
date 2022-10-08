import torch
import torch.nn.functional as F

from conv_core import sudoku, diagonal2

class ROUGELoss(torch.nn.Module):
	def __init__(self, reduction = 'mean', n = 1, max_length=128):
		super(ROUGELoss, self).__init__()
		self.reduction = reduction
		self.n = n
		self.max_length = max_length
	def forward(self, logits, labels):
		logits = logits[:, :self.max_length, :]
		labels = labels[:, :self.max_length]
		labels_attend = (labels != 1).float()
		
		attend = torch.bmm(labels_attend.unsqueeze(2), labels_attend.unsqueeze(1)).unsqueeze(1)
		# overlap = F.embedding(labels.view(-1), F.softmax(logits.view(-1, logits.shape[-1]), dim=-1).T)
		overlap =  attend * torch.stack([F.embedding(b, a.T) for a, b in zip(F.gumbel_softmax(logits), labels)]).unsqueeze(1)
		numerators = - sudoku(overlap).sum((-2, -1))
		# denominators = torch.tensor(overlap.shape[-2:]) - self.n + 1
		denominators = labels_attend.sum(1)
		if self.reduction == 'mean':
			return (numerators / denominators).mean()
		else:
			return overlap
