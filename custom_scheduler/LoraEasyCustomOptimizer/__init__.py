
from typing import Dict, List
from LoraEasyCustomOptimizer.utils import OPTIMIZER

from LoraEasyCustomOptimizer.ademamix import AdEMAMix
from LoraEasyCustomOptimizer.came import CAME
from LoraEasyCustomOptimizer.camefullsr import CAMEFullSR
from LoraEasyCustomOptimizer.compass import Compass, Compass8Bit, Compass8BitBNB
from LoraEasyCustomOptimizer.farmscrop import FARMSCrop
from LoraEasyCustomOptimizer.fcompass import FCompass
from LoraEasyCustomOptimizer.fishmonger import FishMonger, FishMonger8BitBNB
from LoraEasyCustomOptimizer.lpfadamw import LPFAdamW
from LoraEasyCustomOptimizer.rmsprop import RMSProp
from LoraEasyCustomOptimizer.shampoo import ScalableShampoo
from LoraEasyCustomOptimizer.soap import SOAP

OPTIMIZER_LIST: List[OPTIMIZER] = [
    AdEMAMix,
    CAME,
    CAMEFullSR,
    Compass,
    Compass8Bit,
    Compass8BitBNB,
    FARMSCrop,
    FCompass,
    FishMonger,
    FishMonger8BitBNB,
    LPFAdamW,
    RMSProp,
    ScalableShampoo,
    SOAP,
]
OPTIMIZERS: Dict[str, OPTIMIZER] = {str(f"{optimizer.__name__}".lower()): optimizer for optimizer in OPTIMIZER_LIST}