import pandas as pd


class DataLoader:

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Load dataset from Excel file.
        """
        df = pd.read_excel(self.file_path)
        return df

    @staticmethod
    def data_summary(df):
        """
        Print dataset summary.
        """

        print("=" * 60)
        print("Dataset Shape")
        print("=" * 60)
        print(df.shape)

        print("\n" + "=" * 60)
        print("First 5 Rows")
        print("=" * 60)
        print(df.head())

        print("\n" + "=" * 60)
        print("Dataset Info")
        print("=" * 60)
        print(df.info())

        print("\n" + "=" * 60)
        print("Missing Values")
        print("=" * 60)
        print(df.isnull().sum())

        print("\n" + "=" * 60)
        print("Fraud Distribution")
        print("=" * 60)
        print(df["Fraud"].value_counts())

        print("\n" + "=" * 60)
        print("Fraud Percentage")
        print("=" * 60)
        print((df["Fraud"].value_counts(normalize=True) * 100).round(2))