from tqdm import tqdm
from functools import partial
from numpy.random import seed
from multiprocessing import Pool
from multiprocessing import cpu_count
from sklearn.ensemble import IsolationForest

from src.evaluation.evaluation import Evaluation
from src.experiment.adjustment import Correction
from src.experiment.controller import Controller
from src.experiment.method import Method
from src.experiment.setup import Setup
from src.resources.dataset import Dataset
from src.resources.dataset_loader import DataLoader

if __name__ == "__main__":
    seed(1)

    # List Datasets
    experiment_datasets = [Dataset.FRAUD]

    # List Conformal Methods
    experiment_methods = [
        Method.SPLIT_CONFORMAL,
        Method.CV,
        Method.CV_PLUS,
        Method.JACKKNIFE,
        Method.JACKKNIFE_PLUS,
    ]

    for dataset in experiment_datasets:
        print(f"Dataset: {dataset.value}\n-------------------------------------------")

        # Data Import
        dl = DataLoader(dataset=dataset)
        df = dl.df

        # Exclude Jackknife/Jackknife+ for large datasets (n>10_000)
        experiment_methods = [
            method
            for method in experiment_methods
            if dl.num_rows <= 10_000 or method.value not in ["J", "J+"]
        ]

        # Experiment Parameters
        CORES = min(10, cpu_count() - 1)
        J, L = 100, 100  # 100 training sets each with 100 test sets

        inliers = df.loc[df.Class == 0]
        outliers = df.loc[df.Class == 1]

        n_inlier = len(inliers)
        n_train_cal = n_inlier // 2
        n_cal = min(2000, n_train_cal // 2)
        n_test = min(2000, n_train_cal // 3)
        n_test_outlier = n_test // 10  # 10% outliers
        n_test_inlier = n_test - n_test_outlier  # 90% inliers

        params = {}  # default parameters
        model = IsolationForest(**params)

        j = range(J)

        controller = Controller()

        for method in experiment_methods:
            print(f"\nMethod: {method.value}\n-----------------------------")

            setup = Setup(
                n=dl.num_rows,
                L=L,
                model=model,
                inliers=inliers,
                outliers=outliers,
                n_test_inlier=n_test_inlier,
                n_test_outlier=n_test_outlier,
                n_train_cal=n_train_cal,
                n_cal=n_cal,
                correction=Correction.BENJAMINI_HOCHBERG,
                method=method.value
            )

            # Parallel Processing
            func = partial(controller.run_experiment, setup)
            with Pool(CORES) as pool:
                mp_results = list(tqdm(pool.imap_unordered(func, j), total=J))

            # Evaluation
            evaluation = Evaluation(mp_results)
            (
                avg_power,
                avg_fdr,
                q90_power,
                q90_fdr,
                _,
                _,
            ) = evaluation.get_results()
