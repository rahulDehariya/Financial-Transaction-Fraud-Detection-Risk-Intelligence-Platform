from src.services.prediction_service import PredictionService


class BatchPredictionService:

    def __init__(self, model):
        self.service = PredictionService(model)

    def predict(self, dataframe):

        results = []

        for _, row in dataframe.iterrows():

            result = self.service.predict_transaction(
                row.to_dict()
            )

            results.append(result)

        return results