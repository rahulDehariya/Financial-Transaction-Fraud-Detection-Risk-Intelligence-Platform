import joblib
import os


class ModelSaver:

    def __init__(self, model):
        self.model = model

    def save(self, file_name="fraud_model.pkl"):

        os.makedirs("models/trained", exist_ok=True)

        path = os.path.join(
            "models",
            "trained",
            file_name
        )

        joblib.dump(self.model, path)

        print(f"Model saved to {path}")