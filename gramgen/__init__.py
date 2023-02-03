# None attacks
# from .attacks.vanila import VANILA
from .loss import GramGenerateLoss
from .trainer import GGTrainer
from .rouge import RougeScorer



__version__ = '3.3.0'
__all__ = [
    # "CW", "PGDL2", "DeepFool", "PGDRSL2",
    "GramGenerateLoss", "GGTrainer", "RougeScorer",

]
# __wrapper__ = [
#     "LGV", "MultiAttack",
# ]