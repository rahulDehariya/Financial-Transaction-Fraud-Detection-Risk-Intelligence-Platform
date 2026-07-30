import pandas as pd

from src.features.feature_engineering import FeatureEngineering
from src.models.predict import FraudPredictor
from src.risk.risk_scoring import RiskScorer


class PredictionService:

    def __init__(self, model):
        self.predictor = FraudPredictor(model)
        self.risk_engine = RiskScorer(model)

    def predict_transaction(self, transaction: dict):

        # Convert dictionary to DataFrame
        transaction_df = pd.DataFrame([transaction])

        # Feature Engineering
        feature_engineering = FeatureEngineering(transaction_df)

        transaction_df = feature_engineering.create_features()

        # Prediction
        prediction = self.predictor.predict(
            transaction_df
        )
        # Risk Score
        risk = self.risk_engine.score_transaction(
            transaction_df
        )

        return {
            "prediction": prediction["prediction"],
            "fraud_probability": prediction["fraud_probability"],
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
            "recommendation": risk["recommendation"]
        }