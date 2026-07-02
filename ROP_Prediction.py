"""
Rate of Penetration (ROP) prediction using machine learning.

Training data: ROP_DataSet.csv
Blind validation: ROP_Blind_DataSet.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Rate Of Penetration"
FEATURE_COLUMNS = [
    "Hole Depth",
    "Hook Load",
    "Rotary RPM",
    "Rotary Torque",
    "Weight on Bit",
    "Differential Pressure",
    "Gamma at Bit",
]

TRAIN_PATH = Path("ROP_DataSet.csv")
BLIND_PATH = Path("ROP_Blind_DataSet.csv")
MODEL_PATH = Path("rop_model.joblib")
METRICS_PATH = Path("rop_metrics.json")
PLOT_PATH = Path("rop_blind_predictions.png")
CORRELATION_PLOT_PATH = Path("rop_training_correlation.png")
DISTRIBUTION_PLOT_PATH = Path("rop_training_distributions.png")


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing columns: {sorted(missing_cols)}")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return x, y


def build_candidate_models() -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("model", RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("model", GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                )),
            ]
        ),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_percent": mape}


def select_best_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> tuple[str, Pipeline, dict[str, float]]:
    candidates = build_candidate_models()
    cv_scores: dict[str, float] = {}

    print("\nCross-validation on training data (neg RMSE):")
    for name, pipeline in candidates.items():
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        mean_score = float(scores.mean())
        cv_scores[name] = mean_score
        print(f"  {name:20s}: {-mean_score:.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    best_model = candidates[best_name]
    best_model.fit(x_train, y_train)
    return best_name, best_model, cv_scores


def plot_training_correlation(train_df: pd.DataFrame, output_path: Path) -> None:
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    correlation = train_df[columns].corr()
    labels = correlation.columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for row in range(len(labels)):
        for col in range(len(labels)):
            value = correlation.iloc[row, col]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    ax.set_title("Training Dataset: Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_distributions(train_df: pd.DataFrame, output_path: Path) -> None:
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, column in zip(axes, columns):
        values = train_df[column].dropna()
        ax.hist(values, bins=40, color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(values.mean(), color="#C44E52", linestyle="--", linewidth=1.5, label=f"Mean: {values.mean():.2f}")
        ax.axvline(values.median(), color="#55A868", linestyle=":", linewidth=1.5, label=f"Median: {values.median():.2f}")
        ax.set_title(column)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.suptitle("Training Dataset: Distribution of Drilling Features", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.35, edgecolors="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal")
    plt.xlabel("Actual ROP")
    plt.ylabel("Predicted ROP")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_feature_importance(model_name: str, pipeline: Pipeline, feature_names: list[str]) -> None:
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        print(f"\nFeature importance not available for {model_name}.")
        return

    importances = estimator.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    print(f"\nFeature importance ({model_name}):")
    for feature, importance in ranked:
        print(f"  {feature:25s}: {importance:.4f}")


def main() -> None:
    print("Loading datasets...")
    train_df = load_dataset(TRAIN_PATH)
    blind_df = load_dataset(BLIND_PATH)

    x_train, y_train = split_features_target(train_df)
    x_blind, y_blind = split_features_target(blind_df)

    print(f"Training samples:   {len(x_train)}")
    print(f"Blind test samples: {len(x_blind)}")
    print(f"Features:           {', '.join(FEATURE_COLUMNS)}")

    plot_training_correlation(train_df, CORRELATION_PLOT_PATH)
    print(f"Saved correlation plot to: {CORRELATION_PLOT_PATH.resolve()}")

    plot_feature_distributions(train_df, DISTRIBUTION_PLOT_PATH)
    print(f"Saved distribution plot to: {DISTRIBUTION_PLOT_PATH.resolve()}")

    best_name, best_model, cv_scores = select_best_model(x_train, y_train)
    print(f"\nSelected model: {best_name}")

    y_train_pred = best_model.predict(x_train)
    y_blind_pred = best_model.predict(x_blind)

    train_metrics = evaluate_predictions(y_train.to_numpy(), y_train_pred)
    blind_metrics = evaluate_predictions(y_blind.to_numpy(), y_blind_pred)

    print("\nTraining set performance:")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  MAE:  {train_metrics['mae']:.4f}")
    print(f"  R2:   {train_metrics['r2']:.4f}")
    print(f"  MAPE: {train_metrics['mape_percent']:.2f}%")

    print("\nBlind test set performance:")
    print(f"  RMSE: {blind_metrics['rmse']:.4f}")
    print(f"  MAE:  {blind_metrics['mae']:.4f}")
    print(f"  R2:   {blind_metrics['r2']:.4f}")
    print(f"  MAPE: {blind_metrics['mape_percent']:.2f}%")

    print_feature_importance(best_name, best_model, FEATURE_COLUMNS)

    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": best_model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        MODEL_PATH,
    )

    metrics_payload = {
        "selected_model": best_name,
        "cv_neg_rmse": cv_scores,
        "training_metrics": train_metrics,
        "blind_metrics": blind_metrics,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    plot_predictions(
        y_blind.to_numpy(),
        y_blind_pred,
        PLOT_PATH,
        title=f"Blind Test: Actual vs Predicted ROP ({best_name})",
    )

    print(f"\nSaved model to:              {MODEL_PATH.resolve()}")
    print(f"Saved metrics to:            {METRICS_PATH.resolve()}")
    print(f"Saved correlation plot to:   {CORRELATION_PLOT_PATH.resolve()}")
    print(f"Saved distribution plot to:  {DISTRIBUTION_PLOT_PATH.resolve()}")
    print(f"Saved prediction plot to:    {PLOT_PATH.resolve()}")


if __name__ == "__main__":
    main()
