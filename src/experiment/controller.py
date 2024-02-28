from sys import float_info
from sklearn.model_selection import train_test_split

from src.conformal.anomaly_detection import ConformalAnomalyDetection
from src.experiment.setup import Setup
from src.evaluation.tracker import Tracker


class Controller:
    @staticmethod
    def run_experiment(setup: Setup, experiment_id: int):
        tracker = Tracker()

        train, test = train_test_split(
            setup.inliers, test_size=setup.n_train_cal, random_state=experiment_id
        )

        params = {
            "contamination": float_info.min,  # lowest possible value; normal data
            "random_state": experiment_id,
        }
        model = setup.model.set_params(**params)

        cad = ConformalAnomalyDetection(
            setup=setup, train_set=train, test_set=test, detector=model
        )

        cad.calibrate(random_state=experiment_id)

        for l in range(setup.L):
            scores, ground_truth = cad.score(random_state=l)
            p_values = cad.get_marginal_p_values(scores)
            significant_points = cad.control_false_positives(
                p_values, method=setup.correction.value
            )

            tracker.track_power(significant_points, ground_truth, verbose=False)
            tracker.track_fdr(significant_points, ground_truth, verbose=False)

        return tracker
