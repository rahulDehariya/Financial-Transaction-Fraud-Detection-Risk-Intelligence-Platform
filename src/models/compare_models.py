# Compare all models
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class ModelComparison:

    def __init__(self, df, preprocessor):

        self.df = df
        self.preprocessor = preprocessor

    def compare(self):

        X = self.df.drop(columns=["Fraud"])

        y = self.df["Fraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        models = {

            "Logistic Regression": LogisticRegression(max_iter=1000),

            "Random Forest": RandomForestClassifier(
                random_state=42
            )

        }

        results = []

        best_model = None

        best_score = 0

        for name, classifier in models.items():

            pipeline = Pipeline(

                steps=[
                    ("preprocessor", self.preprocessor),
                    ("classifier", classifier)
                ]

            )

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            probabilities = pipeline.predict_proba(X_test)[:, 1]

            score = roc_auc_score(y_test, probabilities)

            results.append({

                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions),
                "Recall": recall_score(y_test, predictions),
                "F1 Score": f1_score(y_test, predictions),
                "ROC AUC": score

            })

            if score > best_score:

                best_score = score

                best_model = pipeline

        return (
            results,
            best_model,
            X_test,
            y_test
        )