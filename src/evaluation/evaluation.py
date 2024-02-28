from statistics import stdev


class Evaluation:
    def __init__(self, mp_results):
        self.sc_avg_fdr = []
        self.sc_avg_power = []

        self.sc_q90_fdr = []
        self.sc_q90_power = []

        for mpr in mp_results:
            avg_power, avg_fdr, q90_power, q90_fdr = mpr.get_performance(verbose=False)
            (
                self.sc_avg_power.append(avg_power),
                self.sc_q90_power.append(q90_power),
                self.sc_avg_fdr.append(avg_fdr),
                self.sc_q90_fdr.append(q90_fdr),
            )

    def get_results(self, verbose=True):
        avg_power = round(sum(self.sc_avg_power) / len(self.sc_avg_power), 3)
        avg_fdr = round(sum(self.sc_avg_fdr) / len(self.sc_avg_fdr), 3)
        q90_power = round(sum(self.sc_q90_power) / len(self.sc_q90_power), 3)
        q90_fdr = round(sum(self.sc_q90_fdr) / len(self.sc_q90_fdr), 3)
        std_power = round(stdev(self.sc_avg_power), 3)
        std_fdr = round(stdev(self.sc_avg_fdr), 3)

        if verbose:
            print(f"Average FDR: {avg_fdr}")
            print(f"90th-Quantile FDR: {q90_fdr}")
            print(f"Standard Deviation FDR: {std_fdr}")
            print("--")
            print(f"Average Power: {avg_power}")
            print(f"90th-Quantile Power: {q90_power}")
            print(f"Standard Deviation Power: {std_power}")

        return (
            avg_power,
            avg_fdr,
            q90_power,
            q90_fdr,
            self.sc_avg_fdr,
            self.sc_avg_power,
        )
