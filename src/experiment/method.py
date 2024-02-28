from enum import Enum


class Method(Enum):
    SPLIT_CONFORMAL: str = "SC"
    CV: str = "CV"
    CV_PLUS: str = "CV+"
    JACKKNIFE: str = "J"
    JACKKNIFE_PLUS: str = "J+"
