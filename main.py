from src.data.data_loader import DataLoader
from src.data.data_validation import DataValidation
from src.features.feature_engineering import FeatureEngineering
from src.preprocessing.preprocessing import DataPreprocessor

DATA_PATH = "data/raw/financial_transactions_fraud_dataset.xlsx"


def main():

    loader = DataLoader(DATA_PATH)

    df = loader.load_data()

    loader.data_summary(df)

    validation = DataValidation(df)

    validation.create_output_directory()

    validation.basic_summary()

    validation.plot_target_distribution()

    validation.plot_transaction_amount()

    validation.plot_correlation()
    feature_engineering = FeatureEngineering(df)
    df = feature_engineering.create_features()

    preprocessor = DataPreprocessor(df)

    transformer = preprocessor.get_preprocessor()

if __name__ == "__main__":
    main()