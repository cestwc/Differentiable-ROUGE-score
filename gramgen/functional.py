from typing import Callable, List, Optional, Tuple, Union
import math
import warnings

import torch
import torch.nn.functional as F

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int

from conv_core import sudoku, diagonal2

Tensor = torch.Tensor

def gram_generate_loss(
    input: Tensor,
    target: Tensor,
    ignore_index: int = 1,
    reduction: str = "mean",
    n: int = 1,
) -> Tensor:

    overlap = batch_embedding(target, discrete_softmax(input.detach(), 1))
    ignored = (target == ignore_index).unsqueeze(-1).repeat(1, 1, overlap.shape[-1]).unsqueeze(1)

    intersection = sudoku(overlap.unsqueeze(1)).long().float().detach()

    idx = 0 == (intersection.sum(dim = -1, keepdim = True).repeat(1, 1, 1, intersection.shape[-1]) + intersection.sum(dim = -2, keepdim = True).repeat(1, 1, intersection.shape[-2], 1))
    intersection[idx] = intersection.shape[0] / (idx.float().sum() + 1e-7)
    intersection = torch.clamp(intersection, 1e-7, 1)
    intersection[ignored] = 0

    
    log_probs = batch_embedding(target, F.log_softmax(input, 1)).unsqueeze(1)
    ret = (- intersection * log_probs ).sum((-3, -2, -1)) + 0.5 * F.cross_entropy(input, torch.ones_like(target) * ignore_index, reduction='none').mean(1)

    if reduction == 'mean':
        return ret.mean()
    elif reduction == 'sum':
        return ret.sum()
    elif reduction == 'intersection':
        return intersection
    elif reduction == 'probs':
        return log_probs.exp()
    else:
        return ret


def batch_embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    r"""A simple lookup table that looks up embeddings in the [batch size] dictionaries.
    """

    return torch.stack([F.embedding(ind, emb, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse) for ind, emb in zip(input, weight)])
 
def discrete_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[DType] = None) -> Tensor:
    r"""Applies a discrete softmax function.

    Softmax is defined as:

    :math:`\text{Discrete Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which discrete softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.

    """
    y_soft = input.softmax(dim, dtype=dtype)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(input, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft
