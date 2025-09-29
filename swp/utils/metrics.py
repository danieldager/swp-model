import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensordict.tensordict import TensorDict


def calc_accuracy(df: pd.DataFrame, error_condition, total_condition) -> float:
    """
    Compute accuracy as 1 - (number of errors / total items) for a given condition.
    """
    total = df.loc[total_condition].shape[0]
    errors = df.loc[error_condition].shape[0]
    return 1 - errors / total


def calc_importance(
    df: pd.DataFrame,
    mode: str = "real",
    y_column: str = "Edit_Distance",
) -> tuple[Pipeline, *tuple[float, ...]]:

    # Prepare data
    df = df.copy()
    match mode:
        case "real":
            df = df[df["Lexicality"] == "real"]
            continuous_features = ["Length", "Zipf_Frequency"]
            categorical_features = ["Morphology"]
        case "both":
            continuous_features = ["Length"]
            categorical_features = ["Lexicality", "Morphology"]
        case "forced":
            continuous_features = ["Length", "Zipf_Frequency"]
            categorical_features = ["Morphology"]
        case _:
            raise ValueError(f"Invalid mode: {mode}, should be 'real' or 'both'.")
    X = df[continuous_features + categorical_features]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", StandardScaler(), continuous_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge())])
    pipeline.fit(X_train, y_train)

    # Permutation (feature) importance
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=100,
        random_state=42,
    )
    # importances_mean is an array of mean importances per feature
    fis = []
    num_fis = len(continuous_features) + len(categorical_features)
    for i in range(num_fis):
        fis.append(result.importances_mean[i])  #  type: ignore

    return pipeline, *fis


def compute_preds(tensordict: TensorDict) -> TensorDict:
    for key in list(tensordict.keys(include_nested=True)):
        match key:
            case (*header, "outputs"):
                tensordict[tuple((*header, "preds"))] = (
                    tensordict[key].detach().argmax(dim=-1)
                )
            case _:
                pass
    return tensordict
