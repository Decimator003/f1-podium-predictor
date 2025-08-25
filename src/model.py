"""
F1 Podium Predictor - Model Module

Trains a classifier on engineered features and predicts podium probabilities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
import joblib

# Target building helpers

def build_training_labels(features: pd.DataFrame) -> Tuple[pd.Series, pd.Index]:
    """
    Build binary labels indicating likely podium candidates.
    Currently uses previous-year indicators as proxy.
    """
    # Use previous-year top3 as weak supervision (placeholder until actual labels available)
    if 'prev_top_3' not in features.columns:
        raise ValueError("Missing 'prev_top_3' in features. Provide labels or previous year flags.")
    y = features['prev_top_3'].astype(int)
    return y, features.index


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select numeric feature columns for model training."""
    exclude = {
        'DriverCode', 'TeamCode',
        'prev_status'
    }
    # Include all numeric features except the label
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['prev_top_3']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    # Add back numeric features only
    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


def build_pipeline(feature_cols: List[str]) -> Pipeline:
    """Create a pipeline with scaling and XGBoost classifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_cols),
        ],
        remainder='drop'
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42,
    )

    pipeline = Pipeline(steps=[('prep', preprocessor), ('clf', model)])
    return pipeline


class PodiumModel:
    def __init__(self, pipeline: Pipeline, feature_cols: List[str]):
        self.pipeline = pipeline
        self.feature_cols = feature_cols

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PodiumModel':
        self.pipeline.fit(X[self.feature_cols], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.feature_cols])[:, 1]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'pipeline': self.pipeline, 'feature_cols': self.feature_cols}, path)

    @staticmethod
    def load(path: str | Path) -> 'PodiumModel':
        obj = joblib.load(path)
        model = PodiumModel(pipeline=obj['pipeline'], feature_cols=obj['feature_cols'])
        return model


def train_model(features: pd.DataFrame) -> Tuple[PodiumModel, Dict[str, Any]]:
    """
    Train podium model using engineered features. Uses previous-year top-3 as proxy labels.
    Returns model and simple metrics (train/val accuracy and logloss).
    """
    y, _ = build_training_labels(features)
    feature_cols = select_feature_columns(features)

    X_train, X_val, y_train, y_val = train_test_split(features, y, test_size=0.25, random_state=42, stratify=y)

    pipeline = build_pipeline(feature_cols)
    model = PodiumModel(pipeline=pipeline, feature_cols=feature_cols)

    model.fit(X_train, y_train)

    # Simple validation metrics
    val_proba = model.predict_proba(X_val)
    val_pred = (val_proba >= 0.5).astype(int)
    metrics = {
        'val_accuracy': float(accuracy_score(y_val, val_pred)),
        'val_logloss': float(log_loss(y_val, val_proba))
    }

    return model, metrics


def predict_top3(model: PodiumModel, features: pd.DataFrame) -> pd.DataFrame:
    """
    Predict podium probabilities and return top-3 drivers with scores.
    """
    proba = model.predict_proba(features)
    out = features[['DriverCode', 'TeamCode']].copy()
    out['podium_probability'] = proba
    out = out.sort_values('podium_probability', ascending=False).head(3).reset_index(drop=True)
    return out


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and run podium predictor')
    parser.add_argument('--features', default=str(Path('data/processed/2025_Hungarian_GP_features.csv')))
    parser.add_argument('--model_out', default=str(Path('models/podium_model.pkl')))
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    model, metrics = train_model(df)
    print('Validation metrics:', metrics)

    Path('models').mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)

    top3 = predict_top3(model, df)
    print('\nTop 3 predictions:')
    print(top3)
