class RiskScorer:

    def __init__(self, model):
        self.model = model

    def score_transaction(self, transaction):

        probability = self.model.predict_proba(
            transaction
        )[0][1]

        risk_score = round(probability * 100)

        if risk_score < 30:

            risk_level = "LOW"

            recommendation = "Approve Transaction"

        elif risk_score < 70:

            risk_level = "MEDIUM"

            recommendation = "Manual Review"

        else:

            risk_level = "HIGH"

            recommendation = "Block Transaction"

        return {

            "fraud_probability": round(probability, 4),

            "risk_score": risk_score,

            "risk_level": risk_level,

            "recommendation": recommendation

        }