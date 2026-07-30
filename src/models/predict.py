class FraudPredictor:

    def __init__(self, model):

        self.model = model

    def predict(self, transaction):

        prediction = self.model.predict(transaction)[0]

        probability = self.model.predict_proba(
            transaction
        )[0][1]

        return prediction, probability