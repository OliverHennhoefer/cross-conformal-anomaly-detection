from scipy import stats
from typing import Union
from copy import copy
from numpy import array, append, sum, median, stack, newaxis, float16
from pandas import DataFrame, concat
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, KFold

from src.experiment.setup import Setup


class ConformalAnomalyDetection:
    def __init__(
        self,
        setup: Setup,
        train_set: DataFrame,
        test_set: DataFrame,
        detector: Union[IsolationForest],
    ):
        self.setup: Setup = setup
        self.method = self.setup.method

        self.train: DataFrame = train_set
        self.test: DataFrame = test_set
        self.detector: Union[IsolationForest] = copy(detector)

        self.n_cal = setup.n_cal
        self.calibration_scores = array([], dtype=float16)
        self.detectors = []

    def calibrate(self, random_state: int) -> None:
        def calibrate_split_conformal():
            train, calib = train_test_split(
                self.train,
                test_size=self.setup.n_cal,
                shuffle=True,
                random_state=random_state,
            )

            train.drop(["Class"], axis=1, inplace=True)
            calib.drop(["Class"], axis=1, inplace=True)

            self.detector.fit(train)
            self.calibration_scores = self.detector.score_samples(calib)

        def calibrate_cross_conformal() -> None:
            def get_n_splits() -> int:
                splits = 20
                if self.method in ["J", "J+"]:
                    splits = len(self.train)
                elif self.setup.n < 10_000:
                    splits = len(self.train) // self.setup.n_cal
                return splits

            n_splits = get_n_splits()
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            train = self.train.drop(["Class"], axis=1)
            for i, (train_index, calib_index) in enumerate(kf.split(train)):
                model = self.detector
                model.set_params(
                    **{"random_state": hash((i, random_state)) % 4294967296}
                )
                model.fit(train.iloc[train_index,])

                if self.method in ["CV+", "J+"]:  # retain oof models
                    self.detectors.append(copy(model))

                self.calibration_scores = append(
                    self.calibration_scores,
                    model.score_samples(train.iloc[calib_index,]),
                )

        if self.method == "SC":
            calibrate_split_conformal()
        elif self.method in ["CV", "J", "CV+", "J+"]:
            calibrate_cross_conformal()

    def score(self, random_state: int):
        test_set = concat(
            [
                self.test.sample(n=self.setup.n_test_inlier, random_state=random_state),
                self.setup.outliers.sample(
                    n=self.setup.n_test_outlier, random_state=random_state
                ),
            ],
            ignore_index=True,
        )

        label = test_set["Class"]
        test_set.drop(["Class"], axis=1, inplace=True)

        if self.method in ["CV+", "J+"]:
            scores_array = stack(
                [model.score_samples(test_set) for model in self.detectors], axis=0
            )
            predictions_array = median(scores_array, axis=0)
        else:
            predictions_array = self.detector.score_samples(test_set)

        return predictions_array, label

    def get_marginal_p_values(self, test_scores):
        p_values_numerator = sum(
            self.calibration_scores <= test_scores[:, newaxis], axis=1
        )
        p_values_marginal = (1.0 + p_values_numerator) / (
            1.0 + len(self.calibration_scores)
        )
        return p_values_marginal

    @staticmethod
    def control_false_positives(p, alpha=0.2, method="BH"):
        if method == "BH":
            return stats.false_discovery_control(p, method="bh") < alpha
        if method == "NONE":
            return p < alpha
