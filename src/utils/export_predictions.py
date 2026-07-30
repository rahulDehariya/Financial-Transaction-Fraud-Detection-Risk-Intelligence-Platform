import os
import pandas as pd


class PredictionExporter:

    @staticmethod
    def save(results):

        os.makedirs(
            "outputs/predictions",
            exist_ok=True
        )

        pd.DataFrame(results).to_csv(

            "outputs/predictions/predictions.csv",

            index=False

        )

        print("Predictions exported.")