import os

import matplotlib.pyplot as plt
import pandas as pd


class DataValidation:

    def __init__(self, df):
        self.df = df

    def basic_summary(self):

        print("=" * 60)
        print("Numerical Statistics")
        print("=" * 60)

        print(self.df.describe())

    def create_output_directory(self):

        os.makedirs("outputs/plots", exist_ok=True)

    def plot_target_distribution(self):

        plt.figure(figsize=(6, 4))

        self.df["Fraud"].value_counts().plot(
            kind="bar"
        )

        plt.title("Fraud Distribution")
        plt.xlabel("Fraud")
        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig("outputs/plots/fraud_distribution.png")

        plt.close()

    def plot_transaction_amount(self):

        plt.figure(figsize=(8, 4))

        self.df["TransactionAmount"].hist(bins=30)

        plt.title("Transaction Amount Distribution")

        plt.xlabel("Amount")

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig("outputs/plots/transaction_amount_distribution.png")

        plt.close()

    def plot_correlation(self):

        numerical_columns = self.df.select_dtypes(
            include=["int64", "float64"]
        )

        correlation = numerical_columns.corr()

        plt.figure(figsize=(10, 6))

        plt.imshow(correlation)

        plt.xticks(
            range(len(correlation.columns)),
            correlation.columns,
            rotation=90
        )

        plt.yticks(
            range(len(correlation.columns)),
            correlation.columns
        )

        plt.colorbar()

        plt.tight_layout()

        plt.savefig("outputs/plots/correlation_matrix.png")

        plt.close()