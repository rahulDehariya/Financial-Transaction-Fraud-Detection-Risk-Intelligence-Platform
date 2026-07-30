import pandas as pd

from src.data.data_loader import DataLoader
from src.data.data_validation import DataValidation
from src.features.feature_engineering import FeatureEngineering
from src.preprocessing.preprocessing import DataPreprocessor
from src.models.compare_models import ModelComparison
from src.models.evaluate import ModelEvaluator

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
    comparison = ModelComparison(df, transformer)
    results, best_model, X_test, y_test = comparison.compare()
    results_df = pd.DataFrame(results)
    evaluator = ModelEvaluator(
        best_model,
        X_test,
        y_test
    )
    evaluator.classification_report()
    evaluator.confusion_matrix()
    evaluator.roc_curve()


if __name__ == "__main__":
    main()