class FraudPredictor:

    def __init__(self, model):
        self.model = model

    def predict(self, transaction_df):

        prediction = self.model.predict(transaction_df)[0]

        probability = self.model.predict_proba(transaction_df)[0][1]

        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability)
        }