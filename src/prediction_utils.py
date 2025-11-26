import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def tune_metadata_regressor(
    factors: pd.DataFrame,
    metadata_target: pd.Series,
    param_grid: Optional[Dict[str, Any]] = None,
    test_size: float = 0.3,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Entrena y optimiza un RandomForestRegressor usando GridSearchCV.
    """
    # 1. Limpieza
    combined = pd.concat([factors, metadata_target], axis=1).dropna()
    X = combined[factors.columns]
    y = combined[metadata_target.name]

    if len(X) < 20:
        return {"error": "Not enough data points for CV"}

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Configurar Grid
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }

    # 4. Grid Search
    print(f"-> Tuning Regressor for {metadata_target.name}...")
    rf = RandomForestRegressor(random_state=random_state)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)

    # 5. Evaluar Mejor Modelo
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    return {
        "model": best_model,
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": factors.columns.tolist(),
    }


def tune_metadata_classifier(
    factors: pd.DataFrame,
    metadata_target: pd.Series,
    param_grid: Optional[Dict[str, Any]] = None,
    test_size: float = 0.3,
    cv: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Entrena y optimiza un RandomForestClassifier usando GridSearchCV.
    """
    # 1. Limpieza
    if not pd.api.types.is_string_dtype(
        metadata_target
    ) and not pd.api.types.is_categorical_dtype(metadata_target):
        metadata_target = metadata_target.astype(str)

    combined = pd.concat([factors, metadata_target], axis=1).dropna()
    X = combined[factors.columns]
    y = combined[metadata_target.name]

    # Filtrar clases pequeñas
    class_counts = y.value_counts()
    valid_classes = class_counts[
        class_counts >= cv + 1
    ].index  # Necesitamos al menos cv+1 muestras
    if len(valid_classes) < 2:
        return {"error": "Not enough classes with sufficient samples for CV"}

    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Configurar Grid
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [10, None],
            "criterion": ["gini", "entropy"],
        }

    # 4. Grid Search
    print(f"-> Tuning Classifier for {metadata_target.name}...")
    rf = RandomForestClassifier(random_state=random_state)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    # 5. Evaluar Mejor Modelo
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    return {
        "model": best_model,
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classes": best_model.classes_,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": factors.columns.tolist(),
    }


def get_factor_importance(model: Any, factor_names: List[str]) -> pd.DataFrame:
    """Helper para extraer importancia de factores."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return pd.DataFrame(
        {
            "Factor": [factor_names[i] for i in indices],
            "Importance": importances[indices],
        }
    )
