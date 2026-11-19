"""Phase 3: the machine learning model.

Owner: ML person

Predicts, for every intersection, the chance of at least one night-time
crime against the person in the next config.LABEL_MONTHS months.

Output files: config.TRAINING_TABLE, config.MODEL_FILE,
              config.NODE_RISK_EVAL, config.NODE_RISK_DEMO
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from saferoutes import config

# A score function takes (true labels, scores) and returns one number.
MetricFn = Callable[[np.ndarray, np.ndarray], float]

# Feature columns, grouped so we can remove one group at a time (ablation).
FEATURE_GROUPS: dict[str, list[str]] = {
    "history": ["pn_1m", "pn_3m", "pn_12m", "pn_decayed"],
    "activity": ["all_12m", "property_12m", "mischief_12m", "theft_12m"],
    "neighbours": ["neighbour_pn_12m"],
    "lighting": ["lights_50m", "avg_lights_per_100m", "min_lights_per_100m"],
    "street_form": ["n_main_roads", "has_path"],
    "season": ["label_quarter"],
    "busyness": ["stops_200m", "sheltered_stops_200m"],   # from stops_clean (config.STOPS_NEAR_NODE_M)
    # future: "night_trips_per_hour" (needs GTFS timetable), "venues_100m" (bars)
}


def all_features(groups: dict[str, list[str]] = FEATURE_GROUPS) -> list[str]:
    """Flatten FEATURE_GROUPS into one list of column names."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Building the training table
# ---------------------------------------------------------------------------
def make_cutoffs(first: date, last: date) -> list[date]:
    """The 1st of every month from `first` to `last`, inclusive."""
    raise NotImplementedError


def build_static_node_features(
    G: nx.MultiDiGraph,
    lights: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Features that don't change over time, one row per intersection.

    Returns columns: node_id, lights_50m, avg_lights_per_100m,
    min_lights_per_100m, n_main_roads, has_path.
    (Street numbers are summarized over the streets meeting at the intersection.)
    """
    raise NotImplementedError


def build_time_features(
    counts: pd.DataFrame,
    G: nx.MultiDiGraph,
    cutoff: date,
) -> pd.DataFrame:
    """Crime-history features for every intersection at ONE cut-off date,
    using only months BEFORE the cut-off.

    Returns columns: node_id, pn_1m, pn_3m, pn_12m, pn_decayed, all_12m,
    property_12m, mischief_12m, theft_12m, neighbour_pn_12m.
    """
    raise NotImplementedError


def build_labels(
    counts: pd.DataFrame,
    cutoff: date,
    label_months: int = config.LABEL_MONTHS,
) -> pd.DataFrame:
    """Label for every intersection at one cut-off.

    Returns columns: node_id, label (1 if any night person-crime in the
    `label_months` after cutoff, else 0), n_future (how many).
    """
    raise NotImplementedError


def build_training_table(
    counts: pd.DataFrame,
    static_features: pd.DataFrame,
    G: nx.MultiDiGraph,
    cutoffs: list[date],
) -> pd.DataFrame:
    """One row per (intersection, cut-off): features + label.

    Returns columns: node_id, cutoff, <all_features()>, label, n_future.
    """
    raise NotImplementedError


def split_by_cutoff(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into (train, validation, test) using the dates in config.

    train:      cutoff <= config.TRAIN_LAST_CUTOFF
    validation: cutoff == config.VALIDATION_CUTOFF
    test:       cutoff == config.TEST_CUTOFF   (use ONCE, at the very end)
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def naive_scores(table: pd.DataFrame) -> np.ndarray:
    """Baseline: score = night person-crimes in the last 12 months (pn_12m)."""
    raise NotImplementedError


def train_logistic(
    train: pd.DataFrame,
    features: list[str],
    C: float = 1.0,
    class_weight: str | None = "balanced",
) -> ClassifierMixin:
    """Logistic regression with log-transformed, scaled inputs."""
    raise NotImplementedError


def train_boosting(
    train: pd.DataFrame,
    features: list[str],
    params: dict[str, float | int] | None = None,
) -> ClassifierMixin:
    """Gradient boosting (sklearn HistGradientBoostingClassifier).

    params may set learning_rate, max_leaf_nodes, min_samples_leaf,
    l2_regularization, max_iter.
    """
    raise NotImplementedError


def predict_proba(model: ClassifierMixin, table: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Probability of label 1 for every row."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def pai_at(scores: np.ndarray, n_future: np.ndarray, top_share: float = 0.05) -> float:
    """Predictive Accuracy Index: share of future crimes in the riskiest
    `top_share` of intersections, divided by `top_share`. 1 = random."""
    raise NotImplementedError


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    n_future: np.ndarray,
    scores_are_probabilities: bool = True,
) -> dict[str, float]:
    """Returns {"auc", "pr_auc", "pai_5", "brier"}.
    "brier" is only included when scores are probabilities."""
    raise NotImplementedError


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: MetricFn,
    n_resamples: int = 1000,
    seed: int = config.RANDOM_SEED,
) -> tuple[float, float]:
    """95% confidence range (low, high) for one metric, by re-sampling rows."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------
def shuffled_label_auc(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    seed: int = config.RANDOM_SEED,
) -> float:
    """Leakage check: retrain with shuffled labels. Should return about 0.5."""
    raise NotImplementedError


def ablation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    groups: dict[str, list[str]] = FEATURE_GROUPS,
) -> pd.DataFrame:
    """Remove one feature group at a time and record the validation scores.

    Returns columns: removed_group, auc, pr_auc, pai_5.
    """
    raise NotImplementedError


def log_run(row: dict[str, object], path: Path = config.RESULTS / "experiment_log.csv") -> None:
    """Append one experiment (date, model, features, settings, scores) to the log."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Calibration, prediction, saving
# ---------------------------------------------------------------------------
def calibrate(
    model: ClassifierMixin,
    validation: pd.DataFrame,
    features: list[str],
) -> ClassifierMixin:
    """Fit a calibration step on the validation set so probabilities are honest."""
    raise NotImplementedError


def predict_node_risk(
    model: ClassifierMixin,
    rows_at_cutoff: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Risk for every intersection at one cut-off.

    Returns columns: node_id, probability, risk_score
    (risk_score = probability / mean probability, so 1 = city average).
    """
    raise NotImplementedError


def save_model(model: ClassifierMixin, path: Path = config.MODEL_FILE) -> None:
    """Save the trained model with joblib."""
    raise NotImplementedError


def load_model(path: Path = config.MODEL_FILE) -> ClassifierMixin:
    """Load a model saved by save_model."""
    raise NotImplementedError
