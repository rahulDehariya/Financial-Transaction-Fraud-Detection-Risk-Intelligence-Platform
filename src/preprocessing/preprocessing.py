from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:

    def __init__(self, df):

        self.df = df

    def get_preprocessor(self):

        X = self.df.drop(columns=["Fraud"])

        numerical_columns = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object"]
        ).columns.tolist()

        # Remove ID columns
        for col in ["TransactionID", "CustomerID", "TransactionDateTime"]:
            if col in categorical_columns:
                categorical_columns.remove(col)

        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numerical_pipeline,
                    numerical_columns
                ),
                (
                    "cat",
                    categorical_pipeline,
                    categorical_columns
                )
            ]
        )

        return preprocessor