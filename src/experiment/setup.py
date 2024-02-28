from dataclasses import dataclass
from typing import Optional, Union
from pandas import DataFrame
from sklearn.ensemble import IsolationForest

from src.experiment.adjustment import Correction


@dataclass
class Setup:
    L: int

    model: Union[IsolationForest]

    n: int
    inliers: DataFrame
    outliers: DataFrame

    n_test_inlier: int
    n_test_outlier: int

    n_train_cal: int
    n_cal: int

    method: Optional[str] = "SC"
    correction: Correction = "BH"
    fold_size: Optional[int] = None
