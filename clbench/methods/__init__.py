from .architecture import PNN
from .base import Method
from .distillation import LwF
from .joint import Joint
from .meta import MER
from .owm import OWM
from .regularization import EWC, SI
from .replay import DGR, ER

METHODS = {
    "finetune": Method,
    "joint": Joint,
    "ewc": EWC,
    "si": SI,
    "lwf": LwF,
    "er": ER,
    "dgr": DGR,
    "owm": OWM,
    "pnn": PNN,
    "mer": MER,
}
