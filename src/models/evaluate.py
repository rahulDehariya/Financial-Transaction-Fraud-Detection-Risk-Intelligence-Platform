import os

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)


class ModelEvaluator:

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

        os.makedirs("outputs/plots", exist_ok=True)
        os.makedirs("outputs/reports", exist_ok=True)

    def classification_report(self):

        predictions = self.model.predict(self.X_test)

        report = classification_report(
            self.y_test,
            predictions
        )

        print(report)

        with open(
            "outputs/reports/classification_report.txt",
            "w"
        ) as file:

            file.write(report)

    def confusion_matrix(self):

        ConfusionMatrixDisplay.from_estimator(
            self.model,
            self.X_test,
            self.y_test
        )

        plt.tight_layout()

        plt.savefig(
            "outputs/plots/confusion_matrix.png"
        )

        plt.close()

    def roc_curve(self):

        RocCurveDisplay.from_estimator(
            self.model,
            self.X_test,
            self.y_test
        )

        plt.tight_layout()

        plt.savefig(
            "outputs/plots/roc_curve.png"
        )

        plt.close()