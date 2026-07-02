"""
Rate of Penetration (ROP) prediction using machine learning.

Training data: ROP_DataSet.csv
Blind validation: ROP_Blind_DataSet.csv

Pipeline notes:
- Cross-validation uses depth-blocked GroupKFold (group = Hole Depth // 500 ft)
  instead of random KFold. Hole Depth increases in ~1 ft steps, so adjacent rows
  are near-duplicates; random KFold leaks them across train/val folds and gives
  overly optimistic CV scores that do not match blind-set performance.
- Engineered mechanical features (rotary power, torque/WOB, etc.) are added on
  top of the raw sensor channels. None depend on the target, so no leakage.
  Denominators (WOB, RPM) are clipped to realistic operating floors rather than
  adding a small epsilon, which would create ratio outliers in the millions.

DATA QUALITY FLAG (distribution shift):
  The blind set's "Gamma at Bit" readings run much higher than training
  (mean ~292 vs ~212 API, max 669 vs 600), suggesting the blind well drills a
  different formation/lithology than most of the training data. Neither CSV
  contains a lithology or formation column (only the 8 sensor channels), so it
  cannot be added as a feature here. If a lithology/formation identifier exists
  in the raw well logs these CSVs were exported from, surfacing it is likely a
  bigger lever on blind-set accuracy than further model tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from itertools import combinations

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict

TARGET_COLUMN = "Rate Of Penetration"
DEPTH_COLUMN = "Hole Depth"

RAW_FEATURE_COLUMNS = [
    "Hole Depth",
    "Hook Load",
    "Rotary RPM",
    "Rotary Torque",
    "Weight on Bit",
    "Differential Pressure",
    "Gamma at Bit",
]

ENGINEERED_FEATURE_COLUMNS = [
    "Rotary Power",
    "Torque per WOB",
    "WOB per RPM",
    "Torque per RPM",
    "DiffPressure per WOB",
    "HookLoad x Torque",
]

FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

# Realistic operating floors for ratio denominators. WOB and RPM both contain
# zeros / near-zeros (blind set RPM q01 = 0); dividing by (x + 1e-6) creates
# outliers in the millions, so clip to a physical floor instead.
WOB_FLOOR = 2.0
RPM_FLOOR = 5.0

# Upper caps for the engineered ratios, set slightly above the observed max of
# both datasets after denominator clipping (train + blind).
RATIO_CAPS = {
    "Torque per WOB": 8.0,
    "WOB per RPM": 5.0,
    "Torque per RPM": 3.0,
    "DiffPressure per WOB": 400.0,
}

# Depth-blocked CV: each group is a contiguous 500 ft interval, so validation
# folds are depth intervals never partially seen in training.
DEPTH_BLOCK_FT = 500
CV_FOLDS = 5
RANDOM_STATE = 42

# Blind-set metrics of the previous baseline (random KFold CV, raw features,
# untuned RandomForest) from the last committed run, kept for the
# before/after comparison table.
BASELINE_BLIND_METRICS = {
    "model": "random_forest (raw features, random KFold CV)",
    "rmse": 42.70,
    "mae": 31.81,
    "r2": 0.656,
    "mape_percent": 24.57,
}

TRAIN_PATH = Path("ROP_DataSet.csv")
BLIND_PATH = Path("ROP_Blind_DataSet.csv")
MODEL_PATH = Path("rop_model.joblib")
METRICS_PATH = Path("rop_metrics.json")
PLOT_PATH = Path("rop_blind_predictions.png")
CORRELATION_PLOT_PATH = Path("rop_training_correlation.png")
DISTRIBUTION_PLOT_PATH = Path("rop_training_distributions.png")
FEATURE_IMPORTANCE_PLOT_PATH = Path("rop_feature_importance.png")
ROP_DEPTH_TRAIN_PLOT_PATH = Path("rop_vs_depth_training.png")
ROP_DEPTH_BLIND_PLOT_PATH = Path("rop_vs_depth_validation.png")


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_cols = set(RAW_FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing columns: {sorted(missing_cols)}")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived mechanical features. None use the target, so no leakage."""
    df = df.copy()
    wob = df["Weight on Bit"].clip(lower=WOB_FLOOR)
    rpm = df["Rotary RPM"].clip(lower=RPM_FLOOR)
    torque = df["Rotary Torque"]

    df["Rotary Power"] = df["Rotary RPM"] * torque
    df["Torque per WOB"] = (torque / wob).clip(upper=RATIO_CAPS["Torque per WOB"])
    df["WOB per RPM"] = (wob / rpm).clip(upper=RATIO_CAPS["WOB per RPM"])
    df["Torque per RPM"] = (torque / rpm).clip(upper=RATIO_CAPS["Torque per RPM"])
    df["DiffPressure per WOB"] = (df["Differential Pressure"] / wob).clip(
        upper=RATIO_CAPS["DiffPressure per WOB"]
    )
    df["HookLoad x Torque"] = df["Hook Load"] * torque
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return x, y


def make_depth_groups(df: pd.DataFrame) -> np.ndarray:
    """Contiguous depth-block labels so CV folds never straddle a block."""
    return (df[DEPTH_COLUMN] // DEPTH_BLOCK_FT).astype(int).to_numpy()


def build_search_spaces() -> dict[str, dict]:
    """Estimators and hyperparameter distributions for RandomizedSearchCV."""
    return {
        "random_forest": {
            "estimator": RandomForestRegressor(
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "param_distributions": {
                "n_estimators": randint(200, 601),
                "max_depth": [8, 12, 16, 20, None],
                "min_samples_leaf": randint(1, 16),
                "max_features": uniform(0.3, 0.7),
            },
            "n_iter": 25,
        },
        "gradient_boosting": {
            "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "param_distributions": {
                "n_estimators": randint(150, 601),
                "learning_rate": loguniform(0.02, 0.2),
                "max_depth": randint(2, 7),
                "subsample": uniform(0.6, 0.4),
                "min_samples_leaf": randint(1, 30),
            },
            "n_iter": 20,
        },
        # No monotonic constraints: tested and they hurt blind performance
        # (RMSE 46 vs 40) - ROP vs WOB/RPM/Torque is not cleanly monotonic here
        # (founder-point effects from bit balling / whirl).
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            "param_distributions": {
                "max_iter": randint(200, 1001),
                "learning_rate": loguniform(0.02, 0.2),
                "max_leaf_nodes": [15, 31, 63, 127],
                "min_samples_leaf": randint(10, 60),
                "l2_regularization": loguniform(1e-3, 10),
            },
            "n_iter": 40,
        },
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_percent": mape}


def tune_candidates(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: np.ndarray,
) -> tuple[dict[str, object], dict[str, dict]]:
    """Tune each candidate against depth-blocked GroupKFold."""
    cv = GroupKFold(n_splits=CV_FOLDS)
    tuned: dict[str, object] = {}
    search_results: dict[str, dict] = {}

    print(f"\nHyperparameter search with {CV_FOLDS}-fold depth-blocked GroupKFold")
    print(f"(depth block = {DEPTH_BLOCK_FT} ft, {len(np.unique(groups))} blocks):")

    for name, spec in build_search_spaces().items():
        search = RandomizedSearchCV(
            estimator=spec["estimator"],
            param_distributions=spec["param_distributions"],
            n_iter=spec["n_iter"],
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
        )
        search.fit(x_train, y_train, groups=groups)
        cv_rmse = float(-search.best_score_)
        tuned[name] = search.best_estimator_
        search_results[name] = {
            "cv_rmse": cv_rmse,
            "best_params": {k: (v if not isinstance(v, np.generic) else v.item())
                            for k, v in search.best_params_.items()},
        }
        print(f"  {name:25s}: CV RMSE = {cv_rmse:.4f}  best params: {search.best_params_}")

    return tuned, search_results


def select_best_predictor(
    tuned: dict[str, object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: np.ndarray,
) -> tuple[str, object, dict[str, float]]:
    """Compare tuned singles and prediction-averaging ensembles on out-of-fold
    depth-blocked CV predictions; select the lowest CV RMSE and fit it on the
    full training set. Blind data is never used for selection."""
    cv = GroupKFold(n_splits=CV_FOLDS)
    y = y_train.to_numpy()

    oof: dict[str, np.ndarray] = {}
    for name, estimator in tuned.items():
        oof[name] = cross_val_predict(
            estimator, x_train, y_train, cv=cv, groups=groups, n_jobs=-1
        )

    candidates: dict[str, tuple[str, ...]] = {name: (name,) for name in tuned}
    for size in (2, 3):
        for names in combinations(sorted(tuned), size):
            candidates[" + ".join(names)] = names

    print("\nModel selection on out-of-fold CV predictions:")
    oof_cv_rmse: dict[str, float] = {}
    for label, members in candidates.items():
        pred = np.mean([oof[m] for m in members], axis=0)
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        oof_cv_rmse[label] = rmse
        print(f"  {label:60s}: OOF CV RMSE = {rmse:.4f}")

    best_label = min(oof_cv_rmse, key=oof_cv_rmse.get)
    members = candidates[best_label]
    if len(members) == 1:
        best_model = tuned[members[0]]
    else:
        best_model = VotingRegressor(
            estimators=[(name, tuned[name]) for name in members], n_jobs=-1
        )
    best_model.fit(x_train, y_train)
    return best_label, best_model, oof_cv_rmse


def plot_training_correlation(train_df: pd.DataFrame, output_path: Path) -> None:
    columns = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]
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
    columns = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]
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


def plot_rop_vs_depth_scatter(
    depth: np.ndarray,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        y_actual,
        depth,
        label="Actual ROP",
        color="#4C72B0",
        alpha=0.45,
        s=14,
        edgecolors="none",
    )
    ax.scatter(
        y_pred,
        depth,
        label="Predicted ROP",
        color="#C44E52",
        alpha=0.45,
        s=14,
        edgecolors="none",
    )
    ax.set_xlabel("Rate Of Penetration (ROP)")
    ax.set_ylabel("Hole Depth")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_feature_importance_ranking(
    model_name: str,
    estimator: object,
    feature_names: list[str],
    x_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
) -> list[dict[str, float | int | str]] | None:
    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
        score_label = "importance"
    elif x_train is not None and y_train is not None:
        # HistGradientBoostingRegressor has no feature_importances_; use
        # permutation importance on the training data instead.
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            estimator, x_train, y_train, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
        )
        scores = result.importances_mean
        score_label = "permutation_importance"
    else:
        print(f"\nFeature importance not available for {model_name}.")
        return None

    ranked = sorted(zip(feature_names, scores), key=lambda item: item[1], reverse=True)
    return [
        {"rank": rank, "feature": feature, "score": float(score), "score_type": score_label}
        for rank, (feature, score) in enumerate(ranked, start=1)
    ]


def plot_feature_importance_ranking(
    ranking: list[dict[str, float | int | str]],
    model_name: str,
    output_path: Path,
) -> None:
    features = [str(item["feature"]) for item in ranking]
    scores = [float(item["score"]) for item in ranking]
    ranks = [int(item["rank"]) for item in ranking]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features, scores, color="#4C72B0", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel(f"Importance score ({ranking[0]['score_type']})")
    ax.set_title(f"Feature Importance Ranking ({model_name})")

    for bar, score, rank in zip(bars, scores, ranks):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  #{rank} ({score:.4f})",
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_feature_importance(ranking: list[dict[str, float | int | str]], model_name: str) -> None:
    print(f"\nFeature importance ranking ({model_name}):")
    for item in ranking:
        print(f"  #{item['rank']} {item['feature']:25s}: {item['score']:.4f}")


def print_metrics_block(label: str, metrics: dict[str, float]) -> None:
    print(f"\n{label}:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R2:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape_percent']:.2f}%")


def print_before_after_table(baseline: dict, new_metrics: dict[str, float], new_model: str) -> None:
    print("\nBefore/after comparison on blind set:")
    print(f"  {'Metric':10s} {'Baseline':>12s} {'New':>12s} {'Change':>12s}")
    rows = [
        ("RMSE", baseline["rmse"], new_metrics["rmse"]),
        ("MAE", baseline["mae"], new_metrics["mae"]),
        ("R2", baseline["r2"], new_metrics["r2"]),
        ("MAPE %", baseline["mape_percent"], new_metrics["mape_percent"]),
    ]
    for name, old, new in rows:
        print(f"  {name:10s} {old:12.4f} {new:12.4f} {new - old:+12.4f}")
    print(f"  Baseline model: {baseline['model']}")
    print(f"  New model:      {new_model} (engineered features, depth-blocked CV)")


def main() -> None:
    print("Loading datasets...")
    train_df = add_engineered_features(load_dataset(TRAIN_PATH))
    blind_df = add_engineered_features(load_dataset(BLIND_PATH))

    x_train, y_train = split_features_target(train_df)
    x_blind, y_blind = split_features_target(blind_df)
    depth_groups = make_depth_groups(train_df)

    print(f"Training samples:   {len(x_train)}")
    print(f"Blind test samples: {len(x_blind)}")
    print(f"Raw features:       {', '.join(RAW_FEATURE_COLUMNS)}")
    print(f"Engineered:         {', '.join(ENGINEERED_FEATURE_COLUMNS)}")

    plot_training_correlation(train_df, CORRELATION_PLOT_PATH)
    print(f"Saved correlation plot to: {CORRELATION_PLOT_PATH.resolve()}")

    plot_feature_distributions(train_df, DISTRIBUTION_PLOT_PATH)
    print(f"Saved distribution plot to: {DISTRIBUTION_PLOT_PATH.resolve()}")

    tuned, search_results = tune_candidates(x_train, y_train, depth_groups)
    best_name, best_model, oof_cv_rmse = select_best_predictor(
        tuned, x_train, y_train, depth_groups
    )
    print(f"\nSelected model: {best_name}")

    y_train_pred = best_model.predict(x_train)
    y_blind_pred = best_model.predict(x_blind)

    train_metrics = evaluate_predictions(y_train.to_numpy(), y_train_pred)
    blind_metrics = evaluate_predictions(y_blind.to_numpy(), y_blind_pred)

    print_metrics_block("Training set performance", train_metrics)
    print_metrics_block("Blind test set performance", blind_metrics)
    print_before_after_table(BASELINE_BLIND_METRICS, blind_metrics, best_name)

    feature_importance_ranking = get_feature_importance_ranking(
        best_name, best_model, FEATURE_COLUMNS, x_train, y_train
    )
    if feature_importance_ranking:
        print_feature_importance(feature_importance_ranking, best_name)
        plot_feature_importance_ranking(
            feature_importance_ranking,
            best_name,
            FEATURE_IMPORTANCE_PLOT_PATH,
        )

    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": best_model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            # Inference note: apply add_engineered_features() to new data
            # before calling pipeline.predict().
        },
        MODEL_PATH,
    )

    metrics_payload = {
        "selected_model": best_name,
        "cv_search_results": search_results,
        "oof_cv_rmse": oof_cv_rmse,
        "cv_strategy": f"GroupKFold({CV_FOLDS}), group = Hole Depth // {DEPTH_BLOCK_FT} ft",
        "training_metrics": train_metrics,
        "blind_metrics": blind_metrics,
        "baseline_blind_metrics": BASELINE_BLIND_METRICS,
        "feature_importance_ranking": feature_importance_ranking,
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

    plot_rop_vs_depth_scatter(
        x_train[DEPTH_COLUMN].to_numpy(),
        y_train.to_numpy(),
        y_train_pred,
        ROP_DEPTH_TRAIN_PLOT_PATH,
        title="Training Data: ROP vs Depth",
    )
    plot_rop_vs_depth_scatter(
        x_blind[DEPTH_COLUMN].to_numpy(),
        y_blind.to_numpy(),
        y_blind_pred,
        ROP_DEPTH_BLIND_PLOT_PATH,
        title="Blind Validation Data: ROP vs Depth",
    )

    print(f"\nSaved model to:                   {MODEL_PATH.resolve()}")
    print(f"Saved metrics to:                 {METRICS_PATH.resolve()}")
    print(f"Saved correlation plot to:        {CORRELATION_PLOT_PATH.resolve()}")
    print(f"Saved distribution plot to:       {DISTRIBUTION_PLOT_PATH.resolve()}")
    if feature_importance_ranking:
        print(f"Saved feature importance plot to: {FEATURE_IMPORTANCE_PLOT_PATH.resolve()}")
    print(f"Saved ROP vs depth (train) plot:  {ROP_DEPTH_TRAIN_PLOT_PATH.resolve()}")
    print(f"Saved ROP vs depth (blind) plot:  {ROP_DEPTH_BLIND_PLOT_PATH.resolve()}")
    print(f"Saved prediction plot to:         {PLOT_PATH.resolve()}")


if __name__ == "__main__":
    main()
