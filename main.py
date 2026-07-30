import pandas as pd

from src.data.data_loader import DataLoader
from src.data.data_validation import DataValidation
from src.features.feature_engineering import FeatureEngineering
from src.preprocessing.preprocessing import DataPreprocessor
from src.models.compare_models import ModelComparison
from src.models.evaluate import ModelEvaluator
from src.models.predict import FraudPredictor
from src.risk.risk_scoring import RiskScorer
from src.services.prediction_service import PredictionService
from src.examples.sample_transaction import get_sample_transaction
from src.models.save_model import ModelSaver

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
    predictor = FraudPredictor(best_model)

    risk_engine = RiskScorer(best_model)

    sample_transaction = X_test.iloc[[0]]

    prediction, probability = predictor.predict(
        sample_transaction
    )

    risk = risk_engine.score_transaction(
        sample_transaction
    )

    print("=" * 60)
    print("Prediction")
    print("=" * 60)

    print(prediction)

    print(probability)

    print()

    print("=" * 60)
    print("Risk Assessment")
    print("=" * 60)

    print(risk)
    service = PredictionService(best_model)
    transaction = get_sample_transaction()
    result = service.predict_transaction(transaction)
    print(result)
    saver = ModelSaver(best_model)
    saver.save()


if __name__ == "__main__":
    main()