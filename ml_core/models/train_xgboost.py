import os
import sys

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine

# Allow imports from ml_core/models when running from project root.
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from cv_utils import purged_time_series_split
from dataset import FEATURES, load_modeling_dataset

# ============================================================
# CONFIGURATION
# ============================================================

DB_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "advanced_gbdt_models"

N_TRIALS = 20
N_CV_SPLITS = 3
PURGE_GAP = 5

INITIAL_TRAIN_FRACTION = 0.60
FINAL_TEST_FRACTION = 0.20

RANDOM_STATE = 42

engine = create_engine(DB_URL)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


# ============================================================
# FINAL HOLDOUT SPLIT
# ============================================================

def create_final_holdout(df):
    """
    Create an untouched chronological final test set.

    The final test set is NEVER used by Optuna.

    Layout:

        DEVELOPMENT DATA | FINAL HOLDOUT
        -----------------|--------------
        80%              | 20%
    """

    split_idx = int(
        len(df) * (1 - FINAL_TEST_FRACTION)
    )

    development_df = df.iloc[:split_idx].copy()
    final_test_df = df.iloc[split_idx:].copy()

    return development_df, final_test_df


# ============================================================
# MODEL
# ============================================================

def create_model(params):
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        **params,
    )


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial, X, y):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            50,
            300,
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            8,
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.005,
            0.1,
            log=True,
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0,
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10,
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-5,
            1.0,
            log=True,
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-3,
            10.0,
            log=True,
        ),
    }

    fold_mse = []

    for fold_number, (train_idx, test_idx) in enumerate(
        purged_time_series_split(
            n_samples=len(X),
            n_splits=N_CV_SPLITS,
            purge_gap=PURGE_GAP,
        ),
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = create_model(params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        predictions = model.predict(X_test)

        mse = mean_squared_error(
            y_test,
            predictions,
        )

        fold_mse.append(mse)

    mean_mse = float(np.mean(fold_mse))

    # Store fold-level information inside Optuna.
    trial.set_user_attr(
        "fold_mse",
        fold_mse,
    )

    return mean_mse


# ============================================================
# MAIN OPTIMIZATION
# ============================================================

def run_optimization(ticker="AAPL"):

    print("=" * 70)
    print("XGBOOST + OPTUNA OPTIMIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load authoritative modeling dataset
    # --------------------------------------------------------

    df = load_modeling_dataset(ticker)

    print(f"\nTicker: {ticker}")
    print(f"Rows: {len(df)}")
    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Separate development data from untouched final test
    # --------------------------------------------------------

    development_df, final_test_df = create_final_holdout(df)

    print(
        f"\nDevelopment rows: "
        f"{len(development_df)}"
    )

    print(
        f"Final holdout rows: "
        f"{len(final_test_df)}"
    )

    print(
        f"Final holdout period: "
        f"{final_test_df['date'].min().date()} → "
        f"{final_test_df['date'].max().date()}"
    )

    X = development_df[FEATURES]
    y = development_df["target_5d"]

    X_final = final_test_df[FEATURES]
    y_final = final_test_df["target_5d"]

    # --------------------------------------------------------
    # Optuna
    # --------------------------------------------------------

    print("\nStarting Optuna optimization...")
    print(f"Trials: {N_TRIALS}")
    print(f"CV folds: {N_CV_SPLITS}")
    print(f"Purge gap: {PURGE_GAP}")

    study = optuna.create_study(
        direction="minimize",
        study_name=f"xgboost_{ticker}",
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X,
            y,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_cv_mse = study.best_value

    print("\n" + "=" * 70)
    print("OPTUNA RESULTS")
    print("=" * 70)

    print(
        f"Best CV MSE: "
        f"{best_cv_mse:.8f}"
    )

    print("\nBest parameters:")

    for key, value in best_params.items():
        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Train final model
    #
    # IMPORTANT:
    # The final holdout has not been touched yet.
    # --------------------------------------------------------

    best_model = create_model(best_params)

    best_model.fit(
        X,
        y,
        verbose=False,
    )

    final_predictions = best_model.predict(
        X_final
    )

    final_mse = mean_squared_error(
        y_final,
        final_predictions,
    )

    final_rmse = np.sqrt(final_mse)

    final_mae = mean_absolute_error(
        y_final,
        final_predictions,
    )

    direction_accuracy = np.mean(
        np.sign(y_final.to_numpy())
        == np.sign(final_predictions)
    )

    correlation = np.corrcoef(
        y_final.to_numpy(),
        final_predictions,
    )[0, 1]

    # --------------------------------------------------------
    # MLflow final run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=f"{ticker}_XGBoost_Optimized"
    ):

        mlflow.log_param(
            "ticker",
            ticker,
        )

        mlflow.log_param(
            "target",
            "5_day_forward_return",
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES),
        )

        mlflow.log_param(
            "n_trials",
            N_TRIALS,
        )

        mlflow.log_param(
            "cv_splits",
            N_CV_SPLITS,
        )

        mlflow.log_param(
            "purge_gap",
            PURGE_GAP,
        )

        mlflow.log_param(
            "final_test_fraction",
            FINAL_TEST_FRACTION,
        )

        mlflow.log_params(
            best_params
        )

        mlflow.log_metric(
            "optuna_cv_mse",
            best_cv_mse,
        )

        mlflow.log_metric(
            "final_mse",
            final_mse,
        )

        mlflow.log_metric(
            "final_rmse",
            final_rmse,
        )

        mlflow.log_metric(
            "final_mae",
            final_mae,
        )

        mlflow.log_metric(
            "direction_accuracy",
            direction_accuracy,
        )

        mlflow.log_metric(
            "correlation",
            correlation,
        )

        mlflow.xgboost.log_model(
            best_model,
            artifact_path="xgboost_model",
        )

    # --------------------------------------------------------
    # Save final predictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "ticker": ticker,
            "date": final_test_df["date"],
            "actual": y_final.to_numpy(),
            "prediction": final_predictions,
        }
    )

    predictions_path = (
        "ml_core/models/"
        f"xgboost_{ticker}_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL XGBOOST RESULTS")
    print("=" * 70)

    print(
        f"MSE                : {final_mse:.8f}"
    )

    print(
        f"RMSE               : {final_rmse:.8f}"
    )

    print(
        f"MAE                : {final_mae:.8f}"
    )

    print(
        f"Direction Accuracy : "
        f"{direction_accuracy:.4f}"
    )

    print(
        f"Correlation        : "
        f"{correlation:.4f}"
    )

    print(
        f"\nSaved predictions to: "
        f"{predictions_path}"
    )

    print("\nXGBoost optimization completed.")


if __name__ == "__main__":
    run_optimization("AAPL")