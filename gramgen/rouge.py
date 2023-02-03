import rouge_score
from rouge_score import rouge_scorer

import torch
import re
from .conv_core import sudoku, diagonal2

def convert_tokens_to_ids(*tokens):
	all_tokens = set().union(*tokens)
	tokens_to_ids = dict(zip(all_tokens, range(len(all_tokens))))
	return [torch.tensor(list(map(tokens_to_ids.get, ts))) for ts in tokens]

def match_ids(ids_1, ids_2):
	num_classes = max(ids_1.max(), ids_2.max()) + 1
	x_1 = torch.nn.functional.one_hot(ids_1, num_classes = num_classes).float()
	x_1.requires_grad = True
	x_2 = torch.nn.functional.one_hot(ids_2, num_classes = num_classes).float()
	x_2.requires_grad = True
	return torch.mm(x_1, x_2.transpose(-1, -2)).unsqueeze(0).unsqueeze(0)

class RougeScorer:
	def __init__(self, rouge_types, use_stemmer=False, split_summaries=False,
				tokenizer=None):
		self.rouge_types = {k:{'precision':0, 'recall':0, 'fmeasure':0} for k in rouge_types}

	def score(self, target, prediction):
		target, prediction = convert_tokens_to_ids(rouge_score.tokenize.tokenize(target, stemmer = None), rouge_score.tokenize.tokenize(prediction, stemmer = None))
		overlap_ = match_ids(target, prediction)

		for rouge_type in self.rouge_types:
			overlap = overlap_
			if re.match(r"rouge[0-9]$", rouge_type):
				n = int(rouge_type[5:])
				for _ in range(n - 1):
					overlap = diagonal2(overlap)
				numerator = sudoku(overlap).sum((-2, -1))
				denominators = torch.tensor(overlap.shape[-2:]) - n + 1
				self.rouge_types[rouge_type]['precision'] = numerator / denominators[0]
				self.rouge_types[rouge_type]['recall'] = numerator / denominators[1]
				self.rouge_types[rouge_type]['fmeasure'] = 2 * numerator / sum(denominators)

		return self.rouge_types
