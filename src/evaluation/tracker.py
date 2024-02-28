from pandas import Series
from numpy import sum, mean, quantile


class Tracker:
    def __init__(self):
        self.power_list = []
        self.fdr_list = []

    def track_power(self, y_hat, y, verbose=True) -> float:
        y_hat = Series(y_hat.astype(int))
        y = y.reset_index(drop=True)

        true_positives = sum(y_hat & y)
        false_negatives = sum(~y_hat & y)

        total_actual_outliers = true_positives + false_negatives

        power = (
            true_positives / total_actual_outliers if total_actual_outliers > 0 else 0
        )

        if verbose:
            print(f"Power: {power}")

        self.power_list.append(power)
        return power

    def track_fdr(self, y_hat, y, verbose=True) -> float:
        y_hat = Series(y_hat.astype(int))
        y = y.reset_index(drop=True)
        try:
            if y_hat.any():
                false_positives = sum(y_hat & ~y)
                true_positives = sum(y_hat & y)
                fdr = false_positives / (false_positives + true_positives)
            elif y.any():
                fdr = 0.0
            else:
                raise ValueError("Inputs don't allow for valid FDR calculation.")

        except ValueError:
            print("FDR could not be calculated. Default of 0.0 was set.")
            fdr = 0.0

        if verbose:
            print(f"FDR: {fdr}")

        self.fdr_list.append(fdr)
        return fdr

    def get_performance(self, verbose=True) -> (float, float, float, float):
        try:
            avg_power = mean(self.power_list)
            avg_fdr = mean(self.fdr_list)

            q90_power = quantile(self.power_list, 0.9)
            q90_fdr = quantile(self.fdr_list, 0.9)

        except (Exception,) as e:
            print(f"Error: {e}. Defaulting to FDR/Power=0.")
            avg_power, avg_fdr, q90_power, q90_fdr = 0, 0, 0, 0

        if verbose:
            print(f"Overall Power: {avg_power}")
            print(f"Overall FDR: {avg_fdr}")

            print(f"90th Percentile Power {q90_power}")
            print(f"90th Percentile FDR {q90_fdr}")

        return avg_power, avg_fdr, q90_power, q90_fdr
