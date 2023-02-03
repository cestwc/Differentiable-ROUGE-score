import torch.nn.functional as F
from .functional import batch_embedding
from .conv_core import sudoku, diagonal2


def compute_metrics(p):
    pred_score, labels = p
    pred_score = F.one_hot(pred_score.argmax(1), pred_score.shape[1]).transpose(1, 2)
    overlap = batch_embedding(labels, pred_score)

    numerator = sudoku(overlap).sum((-3, -2, -1)).detach().numpy()
    denominators = [
        (pred_score.argmax(1) != 1).sum(1).numpy() + 1e-7,
        (p != 1).sum(1).numpy() + 1e-7
    ]

    return {"precision": (numerator / denominators[0]).mean(), "recall": (numerator / denominators[1]).mean(), "eval_f1": (2 * numerator / sum(denominators)).mean()}