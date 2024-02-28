from pandas import DataFrame, read_csv
from src.resources.dataset import Dataset
from src.utils.optimization import reduce_mem_usage


class DataLoader:
    def __init__(self, dataset: Dataset):
        self._df: DataFrame = self.load_data(dataset)
        self._num_rows = self._df.shape[0]

    @staticmethod
    def load_data(dataset: Dataset) -> DataFrame:
        dataset = dataset.value
        df = DataFrame

        if dataset == "thyroid":
            df = read_csv("../data/input/thyroid/thyroid_reduced.csv")
        if dataset == "breast":
            df = read_csv("../data/input/breast/breastw.csv")
        if dataset == "cardio":
            df = read_csv("../data/input/cardio/cardio.csv")
        if dataset == "ionosphere":
            df = read_csv("../data/input/ionosphere/ionosphere.csv")
        if dataset == "musk":
            df = read_csv("../data/input/musk/musk.csv")
        if dataset == "wbc":
            df = read_csv("../data/input/wbc/wbc.csv")
        if dataset == "fraud":
            df = read_csv("../data/input/fraud/fraud.csv")
        if dataset == "shuttle":
            df = read_csv("../data/input/shuttle/shuttle.csv")
        if dataset == "mammography":
            df = read_csv("../data/input/mammography/mammography.csv")
        if dataset == "gamma":
            df = read_csv("../data/input/gamma/gamma.csv")

        else:
            ValueError("Given dataset does not exist.")

        return reduce_mem_usage(df)

    @property
    def df(self):
        return self._df

    @property
    def num_rows(self):
        return self._num_rows
