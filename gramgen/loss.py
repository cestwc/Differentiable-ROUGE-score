from . import functional as F_

import torch
from torch import Tensor
from typing import Callable, Optional

from conv_core import sudoku, diagonal2

class GramGenerateLoss(torch.nn.modules.loss._Loss):

    __constants__ = ['ignore_index', 'reduction', 'n']
    n: int
    ignore_index: int
    # zero_infinity: bool
    # label_smoothing: float

    def __init__(self, ignore_index: int = -100, reduction: str = 'mean', n: int = 1):
        super(GramGenerateLoss, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index
        self.n = n

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F_.gram_generate_loss(input, target, ignore_index=self.ignore_index, reduction=self.reduction, n=self.n)
