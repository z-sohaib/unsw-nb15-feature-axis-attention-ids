from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from unsw_nb15_ids.bundle import DatasetBundle
from unsw_nb15_ids.config import DataConfig
from unsw_nb15_ids.utils import require_file


DROP_COLUMNS = {"id", "attack_cat", "label"}
CATEGORICAL_COLUMNS = ["proto", "service", "state"]


def load_unsw_nb15(config: DataConfig, seed: int) -> DatasetBundle:
    train_path = require_file(config.train_path, "UNSW-NB15 training CSV")
    test_path = require_file(config.test_path, "UNSW-NB15 testing CSV")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train_full, class_names = _labels(train_df, config.task)
    y_test, _ = _labels(test_df, config.task, class_names=class_names)

    x_train_full_df = train_df.drop(columns=[c for c in DROP_COLUMNS if c in train_df.columns])
    x_test_df = test_df.drop(columns=[c for c in DROP_COLUMNS if c in test_df.columns])

    train_indices, val_indices = _split_indices(train_df, config, seed)
    x_train_df = x_train_full_df.iloc[train_indices].copy()
    x_val_df = x_train_full_df.iloc[val_indices].copy()
    y_train = y_train_full[train_indices]
    y_val = y_train_full[val_indices]

    preprocessor = _build_preprocessor(x_train_df, log_transform=config.log_transform)
    preprocessor.fit(x_train_df)

    if config.save_preprocessor:
        _save_preprocessor(preprocessor, config, x_train_df)

    if config.apply_smote and config.smote_method == "smotenc":
        x_train_df, y_train = _apply_smotenc(
            x_train_df,
            y_train,
            seed,
            target_count=config.smote_target_count,
            target_counts=config.smote_target_counts,
            class_names=class_names,
        )

    x_train = preprocessor.transform(x_train_df)
    x_val = preprocessor.transform(x_val_df)
    x_test = preprocessor.transform(x_test_df)

    x_train = _to_dense_float32(x_train)
    x_val = _to_dense_float32(x_val)
    x_test = _to_dense_float32(x_test)

    if config.apply_smote and config.smote_method == "encoded_smote":
        x_train, y_train = _apply_smote(
            x_train,
            y_train,
            seed,
            target_count=config.smote_target_count,
            target_counts=config.smote_target_counts,
            class_names=class_names,
        )
    elif config.apply_smote and config.smote_method not in {"smotenc", "encoded_smote"}:
        raise ValueError(
            "Unsupported SMOTE method. Use 'smotenc' for categorical-safe SMOTE "
            "or 'encoded_smote' only for explicit legacy experiments."
        )

    return DatasetBundle(
        x_train=_as_sequence(x_train),
        y_train=y_train.astype(np.int64),
        x_val=_as_sequence(x_val),
        y_val=y_val.astype(np.int64),
        x_test=_as_sequence(x_test),
        y_test=y_test.astype(np.int64),
        class_names=class_names,
    )


def _labels(
    frame: pd.DataFrame,
    task: str,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if task == "binary":
        if "label" not in frame.columns:
            raise ValueError("UNSW-NB15 binary task requires a 'label' column.")
        return frame["label"].astype(int).to_numpy(), ["normal", "attack"]

    if task != "multiclass":
        raise ValueError(f"Unsupported UNSW-NB15 task: {task}")
    if "attack_cat" not in frame.columns:
        raise ValueError("UNSW-NB15 multiclass task requires an 'attack_cat' column.")

    labels = frame["attack_cat"].fillna("Normal").astype(str).str.strip()
    if class_names is None:
        class_names = sorted(labels.unique().tolist())
    mapping = {name: idx for idx, name in enumerate(class_names)}
    unknown = sorted(set(labels.unique()) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown attack categories in split: {unknown}")
    return labels.map(mapping).astype(int).to_numpy(), class_names


def _build_preprocessor(frame: pd.DataFrame, log_transform: bool = False) -> ColumnTransformer:
    categorical = [c for c in CATEGORICAL_COLUMNS if c in frame.columns]
    numeric = [c for c in frame.columns if c not in categorical]
    numeric_steps = []
    if log_transform:
        numeric_steps.append(
            ("log1p", FunctionTransformer(_log1p_nonnegative, feature_names_out="one-to-one"))
        )
    numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", Pipeline(numeric_steps), numeric),
        ],
        remainder="drop",
    )


def _split_indices(
    train_df: pd.DataFrame,
    config: DataConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if config.split_indices_path and config.split_indices_path.exists():
        saved = np.load(config.split_indices_path)
        train_indices = saved["train_indices"].astype(int)
        val_indices = saved["validation_indices"].astype(int)
        _validate_saved_indices(train_indices, val_indices, len(train_df), config.split_indices_path)
        return train_indices, val_indices

    stratify_labels, _ = _labels(train_df, "multiclass" if "attack_cat" in train_df.columns else config.task)
    indices = np.arange(len(train_df))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=config.validation_size,
        stratify=stratify_labels,
        random_state=seed,
    )
    train_indices = np.sort(train_indices.astype(int))
    val_indices = np.sort(val_indices.astype(int))

    if config.split_indices_path:
        config.split_indices_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.split_indices_path,
            train_indices=train_indices,
            validation_indices=val_indices,
            seed=np.array([seed], dtype=np.int64),
            validation_size=np.array([config.validation_size], dtype=np.float64),
        )
    return train_indices, val_indices


def _validate_saved_indices(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    total_rows: int,
    path: object,
) -> None:
    if len(set(train_indices.tolist()) & set(val_indices.tolist())) > 0:
        raise ValueError(f"Saved split indices overlap: {path}")
    if len(train_indices) + len(val_indices) != total_rows:
        raise ValueError(f"Saved split indices do not cover the official training split: {path}")
    if train_indices.min(initial=0) < 0 or val_indices.min(initial=0) < 0:
        raise ValueError(f"Saved split indices contain negative values: {path}")
    if train_indices.max(initial=0) >= total_rows or val_indices.max(initial=0) >= total_rows:
        raise ValueError(f"Saved split indices exceed official training rows: {path}")


def _log1p_nonnegative(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.log1p(np.clip(array, a_min=0.0, a_max=None))


def _save_preprocessor(preprocessor: ColumnTransformer, config: DataConfig, frame: pd.DataFrame) -> None:
    if config.preprocessor_path is None:
        return
    config.preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "preprocessor": preprocessor,
            "metadata": {
                "task": config.task,
                "log_transform": config.log_transform,
                "feature_columns_before_encoding": list(frame.columns),
                "categorical_columns": [c for c in CATEGORICAL_COLUMNS if c in frame.columns],
                "split_indices_path": str(config.split_indices_path)
                if config.split_indices_path
                else None,
            },
        },
        config.preprocessor_path,
    )


def _to_dense_float32(values: object) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def _apply_smote(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    target_count: int | None = None,
    target_counts: dict[str, int] | None = None,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y_train)
    minority_count = int(counts[counts > 0].min())
    if minority_count < 2:
        raise ValueError("SMOTE requires at least two samples in every present class.")
    k_neighbors = min(5, minority_count - 1)
    sampling_strategy = _smote_sampling_strategy(
        y_train,
        target_count=target_count,
        target_counts=target_counts,
        class_names=class_names,
    )
    if sampling_strategy is None:
        return x_train.astype(np.float32), y_train.astype(np.int64)
    smote = SMOTE(
        random_state=seed,
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
    )
    x_balanced, y_balanced = smote.fit_resample(x_train, y_train)
    return x_balanced.astype(np.float32), y_balanced.astype(np.int64)


def _apply_smotenc(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    target_count: int | None = None,
    target_counts: dict[str, int] | None = None,
    class_names: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    categorical = [c for c in CATEGORICAL_COLUMNS if c in x_train.columns]
    if not categorical:
        x_balanced, y_balanced = _apply_smote(
            x_train.to_numpy(dtype=np.float32),
            y_train,
            seed,
            target_count=target_count,
            target_counts=target_counts,
            class_names=class_names,
        )
        return pd.DataFrame(x_balanced, columns=x_train.columns), y_balanced

    encoded = x_train.copy()
    category_values: dict[str, list[str]] = {}
    for column in categorical:
        values = sorted(encoded[column].astype(str).unique().tolist())
        category_values[column] = values
        mapping = {value: idx for idx, value in enumerate(values)}
        encoded[column] = encoded[column].astype(str).map(mapping).astype(float)

    for column in encoded.columns:
        if column not in categorical:
            encoded[column] = pd.to_numeric(encoded[column], errors="raise").astype(float)

    counts = np.bincount(y_train)
    minority_count = int(counts[counts > 0].min())
    if minority_count < 2:
        raise ValueError("SMOTENC requires at least two samples in every present class.")
    k_neighbors = min(5, minority_count - 1)
    categorical_indices = [encoded.columns.get_loc(column) for column in categorical]
    sampling_strategy = _smote_sampling_strategy(
        y_train,
        target_count=target_count,
        target_counts=target_counts,
        class_names=class_names,
    )
    if sampling_strategy is None:
        return x_train, y_train.astype(np.int64)
    smote = SMOTENC(
        categorical_features=categorical_indices,
        random_state=seed,
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
    )
    balanced_array, y_balanced = smote.fit_resample(encoded.to_numpy(dtype=np.float64), y_train)
    balanced = pd.DataFrame(balanced_array, columns=encoded.columns)

    for column in categorical:
        categories = category_values[column]
        codes = np.rint(balanced[column].to_numpy(dtype=float)).astype(int)
        codes = np.clip(codes, 0, len(categories) - 1)
        balanced[column] = [categories[code] for code in codes]
    for column in balanced.columns:
        if column not in categorical:
            balanced[column] = pd.to_numeric(balanced[column], errors="raise").astype(float)

    return balanced[x_train.columns], y_balanced.astype(np.int64)


def _smote_sampling_strategy(
    y_train: np.ndarray,
    target_count: int | None,
    target_counts: dict[str, int] | None,
    class_names: list[str] | None,
) -> str | dict[int, int] | None:
    if target_counts:
        if class_names is None:
            raise ValueError("Per-class SMOTE targets require class_names.")
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        unknown = sorted(set(target_counts) - set(name_to_idx))
        if unknown:
            raise ValueError(f"Unknown SMOTE target class names: {unknown}")
        counts = np.bincount(y_train, minlength=len(class_names))
        strategy = {
            name_to_idx[class_name]: int(target)
            for class_name, target in target_counts.items()
            if 0 < counts[name_to_idx[class_name]] < int(target)
        }
        return strategy or None
    if target_count is None:
        return "auto"
    counts = np.bincount(y_train)
    strategy = {
        class_idx: int(target_count)
        for class_idx, count in enumerate(counts)
        if 0 < count < target_count
    }
    if not strategy:
        return None
    return strategy


def _as_sequence(x: np.ndarray) -> np.ndarray:
    return np.expand_dims(x, axis=1).astype(np.float32)
