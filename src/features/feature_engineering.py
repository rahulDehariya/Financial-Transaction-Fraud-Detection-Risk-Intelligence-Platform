import pandas as pd


class FeatureEngineering:

    def __init__(self, df):
        self.df = df.copy()

    def create_features(self):

        # Convert datetime
        self.df["TransactionDateTime"] = pd.to_datetime(
            self.df["TransactionDateTime"]
        )

        # Transaction Hour
        self.df["TransactionHour"] = (
            self.df["TransactionDateTime"].dt.hour
        )

        # Night Transaction
        self.df["IsNightTransaction"] = (
            self.df["TransactionHour"]
            .between(0, 5)
            .astype(int)
        )

        # High Amount Transaction
        self.df["HighAmountTransaction"] = (
            self.df["TransactionAmount"] > 3000
        ).astype(int)

        # Balance Ratio
        self.df["BalanceToAmountRatio"] = (
            self.df["AccountBalance"]
            /
            (self.df["TransactionAmount"] + 1)
        )

        return self.df