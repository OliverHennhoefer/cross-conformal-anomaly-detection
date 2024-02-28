from enum import Enum


class Correction(Enum):
    BENJAMINI_HOCHBERG: str = "BH"
    NO_ADJUSTMENT: str = "NONE"
