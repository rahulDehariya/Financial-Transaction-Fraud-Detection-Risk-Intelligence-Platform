from src.data.data_loader import DataLoader


DATA_PATH = "data/raw/financial_transactions_fraud_dataset.xlsx"


def main():

    loader = DataLoader(DATA_PATH)

    df = loader.load_data()

    loader.data_summary(df)


if __name__ == "__main__":
    main()