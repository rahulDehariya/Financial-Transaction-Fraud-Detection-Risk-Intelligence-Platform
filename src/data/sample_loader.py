import pandas as pd


class SampleTransactionLoader:

    @staticmethod
    def load():

        return pd.read_csv(
            "data/sample_transactions.csv"
        )