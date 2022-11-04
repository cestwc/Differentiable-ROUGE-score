import torch
import torch.nn.functional as F

from conv_core import sudoku, diagonal2

def discrete_softmax(logits, dim = -1):
	y_soft = logits.softmax(dim)
	index = y_soft.max(dim, keepdim=True)[1]
	y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
	return y_hard - y_soft.detach() + y_soft
	
class ROUGELoss(torch.nn.Module):
	def __init__(self, reduction = 'mean', n = 1, max_length=128):
		super(ROUGELoss, self).__init__()
		self.reduction = reduction
		self.n = n
		self.max_length = max_length
	def forward(self, logits, labels):
		# logits = logits[:, :self.max_length, :]
		# labels = labels[:, :self.max_length]
		# labels_attend = (labels != 1).float()

		# attend = torch.bmm(labels_attend.unsqueeze(2), labels_attend.unsqueeze(1)).unsqueeze(1)
		# overlap = F.embedding(labels.view(-1), F.softmax(logits.view(-1, logits.shape[-1]), dim=-1).T)
		overlap =  torch.stack([F.embedding(b, a.T) for a, b in zip(discrete_softmax(logits, -1), labels)]).unsqueeze(1)
		overlap = sudoku(overlap).long().float().detach()
		# for k in range(len(overlap)):
		# 	h = overlap[k].sum(-2).squeeze()
		# 	v = overlap[k].sum(-1).squeeze()
		# 	for kh in range(len(h)):
		# 		for kv in range(len(v)):
		# 			if h[kh] == 0 and v[kv] == 0:
		# 				overlap[k, 0, kv, kh] = 0.5
		idx = 0 == (overlap.sum(dim = -1, keepdim = True).repeat(1, 1, 1, overlap.shape[-1]) + overlap.sum(dim = -2, keepdim = True).repeat(1, 1, overlap.shape[-2], 1))
		overlap[idx] = 0.5
		overlap = torch.clamp(overlap, 1e-1, 1)

		probs =  torch.stack([F.embedding(b, a.T) for a, b in zip(F.softmax(logits, -1), labels)]).unsqueeze(1)
		numerators = - (overlap * probs).sum((-2, -1)) * 2
		denominators = sum(overlap.shape[-2:]) - self.n + 1


		# denominators = labels_attend.sum(1)
		if self.reduction == 'mean':
			return 1 + (numerators / denominators).mean()
		else:
			return overlap
