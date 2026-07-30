import joblib


class ModelLoader:

    @staticmethod
    def load(path="models/trained/fraud_model.pkl"):

        return joblib.load(path)