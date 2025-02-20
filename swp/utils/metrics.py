import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def calc_accuracy(df: pd.DataFrame, error_condition, total_condition) -> float:
    """
    Compute accuracy as 1 - (number of errors / total items) for a given condition.
    """
    total = df.loc[total_condition].shape[0]
    errors = df.loc[error_condition].shape[0]
    return 1 - errors / total


def calc_importance(
    df: pd.DataFrame, mode: str = "real"
) -> tuple[Pipeline, float, float, float]:

    df = df.copy()

    # Define features: include the continuous variables and the categorical one.
    if mode == "real":
        df = df[df["Lexicality"] == "real"]
        continuous_features = ["Length", "Zipf Frequency"]
        categorical_features = ["Morphology"]
    elif mode == "both":
        continuous_features = ["Length"]
        categorical_features = ["Lexicality", "Morphology"]
    else:
        raise ValueError(f"Invalid mode: {mode}, should be 'real' or 'both'.")

    X = df[continuous_features + categorical_features]
    y = df["Edit Distance"]

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a preprocessor that standardizes continuous features and one-hot encodes categorical features.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", StandardScaler(), continuous_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )

    # Build a pipeline with the preprocessor and a linear regression model.
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )
    pipeline.fit(X_train, y_train)

    # Compute permutation importance on the test set.
    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=100, random_state=42, scoring="r2"
    )
    # result.importances_mean gives an array with the mean importance per feature.
    # The order corresponds to continuous_features: "Length" then "Zipf Frequency"
    fi1 = result.importances_mean[0]  # type: ignore
    fi2 = result.importances_mean[1]  # type: ignore
    fi3 = result.importances_mean[2]  # type: ignore

    return pipeline, fi1, fi2, fi3
