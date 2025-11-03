 
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import warnings


# --------------------------- Data structures (СПОЧАТКУ!) ---------------------------

@dataclass(frozen=True)
class DatasetBundle:
    """Контейнер з розбитими та перетвореними наборами даних."""
    train_X: pd.DataFrame
    train_y: pd.Series
    val_X: pd.DataFrame
    val_y: pd.Series
    test_X: pd.DataFrame
    test_y: pd.Series
    numeric_cols: List[str]
    categorical_cols: List[str]
    encoded_cols: List[str]


# --------------------------- Single-purpose helpers ---------------------------

def resolve_data_path(csv_name: str) -> Path:
    """
    Повертає абсолютний шлях до CSV відносно місця розташування цього файла-модуля.
    """
    return (Path(__file__).resolve().parent / csv_name).resolve()


def read_csv_at(path: Path, index_col: int = 0) -> pd.DataFrame:
    """
    Зчитує CSV у DataFrame.
    """
    return pd.read_csv(path, index_col=index_col)


def drop_required_na(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """
    Видаляє рядки з NaN у критично важливих колонках.
    """
    return df.dropna(subset=required_cols).copy()


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Розділяє датафрейм на X (ознаки) та y (таргет).
    """
    input_cols = [c for c in df.columns if c not in set(drop_cols + [target_col])]
    X = df[input_cols].copy()
    y = df[target_col].astype(int)
    return X, y


def split_60_20_20(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Ділить дані у пропорції 60/20/20 для train/val/test зі стратифікацією по y.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Визначає списки числових та категоріальних колонок у датафреймі ознак.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    return numeric_cols, categorical_cols


def fit_numeric_imputer(X_train: pd.DataFrame, numeric_cols: List[str]) -> SimpleImputer:
    """
    Навчає імп'ютер на числових колонках тренувального набору.
    """
    return SimpleImputer(strategy="mean").fit(X_train[numeric_cols]) if numeric_cols else SimpleImputer(strategy="mean")


def transform_numeric_imputer(
    imputer: SimpleImputer,
    df: pd.DataFrame,
    numeric_cols: List[str]
) -> pd.DataFrame:
    """
    Застосовує імп'ютер до числових колонок і повертає копію df.
    """
    out = df.copy()
    if numeric_cols:
        out[numeric_cols] = imputer.transform(out[numeric_cols])
    return out


def fit_scaler(X_train: pd.DataFrame, numeric_cols: List[str]) -> MinMaxScaler:
    """
    Навчає MinMaxScaler на числових колонках тренувального набору.
    """
    return MinMaxScaler().fit(X_train[numeric_cols]) if numeric_cols else MinMaxScaler()


def transform_scaler(
    scaler: MinMaxScaler,
    df: pd.DataFrame,
    numeric_cols: List[str]
) -> pd.DataFrame:
    """
    Застосовує масштабування до числових колонок і повертає копію df.
    """
    out = df.copy()
    if numeric_cols:
        out[numeric_cols] = scaler.transform(out[numeric_cols])
    return out


def fit_onehot_encoder(X_train: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """
    Навчає OneHotEncoder на категоріальних колонках тренувального набору.
    Сумісно з різними версіями scikit-learn (sparse_output vs sparse).
    """
    if not categorical_cols:
        try:
            return OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(pd.DataFrame(index=X_train.index))
        except TypeError:
            return OneHotEncoder(sparse=False, handle_unknown="ignore").fit(pd.DataFrame(index=X_train.index))

    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X_train[categorical_cols])
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore").fit(X_train[categorical_cols])


def transform_onehot_concat(
    encoder: OneHotEncoder,
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Застосовує one-hot кодування до категоріальних колонок і повертає
    новий df із закодованими ознаками, а також список назв згенерованих колонок.
    """
    if not categorical_cols:
        return df.copy(), []

    encoded = encoder.transform(df[categorical_cols])
    encoded_col_names = list(encoder.get_feature_names_out(categorical_cols))
    encoded_df = pd.DataFrame(encoded, index=df.index, columns=encoded_col_names)
    out = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    return out, encoded_col_names


# --------------------------- Orchestration ---------------------------

def prepare_datasets_from_df(
    df: pd.DataFrame,
    target_col: str = "Exited",
    required_cols: List[str] = None,
    random_state: int = 42
) -> DatasetBundle:
    """
    Версія prepare_datasets, що приймає DataFrame замість імені файлу.
    """
    if required_cols is None:
        required_cols = ["CustomerId", target_col]

    raw_df = drop_required_na(df, required_cols)
    X, y = split_features_target(raw_df, target_col=target_col, drop_cols=["CustomerId"])
    X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20(X, y, random_state=random_state)
    
    numeric_cols, categorical_cols = detect_column_types(X_train)
    
    num_imputer = fit_numeric_imputer(X_train, numeric_cols)
    X_train = transform_numeric_imputer(num_imputer, X_train, numeric_cols)
    X_val = transform_numeric_imputer(num_imputer, X_val, numeric_cols)
    X_test = transform_numeric_imputer(num_imputer, X_test, numeric_cols)
    
    scaler = fit_scaler(X_train, numeric_cols)
    X_train = transform_scaler(scaler, X_train, numeric_cols)
    X_val = transform_scaler(scaler, X_val, numeric_cols)
    X_test = transform_scaler(scaler, X_test, numeric_cols)
    
    encoder = fit_onehot_encoder(X_train, categorical_cols)
    X_train, encoded_cols = transform_onehot_concat(encoder, X_train, categorical_cols)
    X_val, _ = transform_onehot_concat(encoder, X_val, categorical_cols)
    X_test, _ = transform_onehot_concat(encoder, X_test, categorical_cols)
    
    return DatasetBundle(
        train_X=X_train, train_y=y_train,
        val_X=X_val, val_y=y_val,
        test_X=X_test, test_y=y_test,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        encoded_cols=encoded_cols,
    )


def prepare_datasets(
    csv_name: str = "train.csv",
    target_col: str = "Exited",
    required_cols: List[str] = None,
    random_state: int = 42
) -> DatasetBundle:
    """Оригінальна функція для читання з CSV."""
    if required_cols is None:
        required_cols = ["CustomerId", target_col]

    data_path = resolve_data_path(csv_name)
    raw_df = read_csv_at(data_path, index_col=0)
    raw_df = drop_required_na(raw_df, required_cols)

    X, y = split_features_target(raw_df, target_col=target_col, drop_cols=["CustomerId"])
    X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20(X, y, random_state=random_state)
    
    numeric_cols, categorical_cols = detect_column_types(X_train)
    
    num_imputer = fit_numeric_imputer(X_train, numeric_cols)
    X_train = transform_numeric_imputer(num_imputer, X_train, numeric_cols)
    X_val = transform_numeric_imputer(num_imputer, X_val, numeric_cols)
    X_test = transform_numeric_imputer(num_imputer, X_test, numeric_cols)
    
    scaler = fit_scaler(X_train, numeric_cols)
    X_train = transform_scaler(scaler, X_train, numeric_cols)
    X_val = transform_scaler(scaler, X_val, numeric_cols)
    X_test = transform_scaler(scaler, X_test, numeric_cols)
    
    encoder = fit_onehot_encoder(X_train, categorical_cols)
    X_train, encoded_cols = transform_onehot_concat(encoder, X_train, categorical_cols)
    X_val, _ = transform_onehot_concat(encoder, X_val, categorical_cols)
    X_test, _ = transform_onehot_concat(encoder, X_test, categorical_cols)
    
    return DatasetBundle(
        train_X=X_train, train_y=y_train,
        val_X=X_val, val_y=y_val,
        test_X=X_test, test_y=y_test,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        encoded_cols=encoded_cols,
    )

def prepare_submission_test(
    raw_df_test: pd.DataFrame,
    train_data: DatasetBundle
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Підготовка submission test даних.
    Застосовує ті ж трансформації, що були на train.
    
    Returns:
        X_test: Оброблені features
        ids: CustomerId для submission файлу
    """
    # ---------------------- Визначаємо колонку з ID ----------------------
    id_col = 'CustomerId' if 'CustomerId' in raw_df_test.columns else 'id'
    ids = raw_df_test[id_col].copy()
    X_test = raw_df_test.drop(columns=[id_col]).copy()

    # ---------------------- Заповнення NaN числових колонок ----------------------
    numeric_cols = train_data.numeric_cols
    if numeric_cols:
        X_test[numeric_cols] = train_data.num_imputer.transform(X_test[numeric_cols])

    # ---------------------- Масштабування числових колонок ----------------------
    if numeric_cols:
        X_test[numeric_cols] = train_data.scaler.transform(X_test[numeric_cols])

    # ---------------------- One-Hot Encoding для категоріальних колонок ----------------------
    categorical_cols = train_data.categorical_cols
    if categorical_cols:
        encoded = train_data.encoder.transform(X_test[categorical_cols])
        encoded_cols = list(train_data.encoder.get_feature_names_out(categorical_cols))
        encoded_df = pd.DataFrame(encoded, index=X_test.index, columns=encoded_cols)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), encoded_df], axis=1)

    # ---------------------- Додавання відсутніх колонок ----------------------
    missing_cols = set(train_data.train_X.columns) - set(X_test.columns)
    if missing_cols:
        zeros_df = pd.DataFrame(0, index=X_test.index, columns=missing_cols)
        X_test = pd.concat([X_test, zeros_df], axis=1)

    # ---------------------- Видалення зайвих колонок ----------------------
    extra_cols = set(X_test.columns) - set(train_data.train_X.columns)
    if extra_cols:
        X_test = X_test.drop(columns=list(extra_cols))

    # ---------------------- Впорядкування колонок ----------------------
    X_test = X_test[train_data.train_X.columns]

    return X_test, ids
