# from pathlib import Path
# import pandas as pd
# import numpy as np
#
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
#
#
# def process_data(csv_name: str = "train.csv"):
#     """
#     Читає дані, чистить, ділить на train/val/test, імп'ютить, масштабує числові
#     та one-hot кодує категоріальні ознаки. Повертає словник з наборами даних.
#     """
#     # 1) Читаємо CSV відносно поточного файлу модуля (стабільніше за CWD)
#     data_path = Path(__file__).resolve().parent / csv_name
#     raw_df = pd.read_csv(data_path, index_col=0)
#
#     # 2) Прибираємо рядки з пропусками у ключових колонках
#     raw_df = raw_df.dropna(subset=["CustomerId", "Exited"]).copy()
#
#     # 3) Визначаємо колонки ознак і таргет
#     target_col = "Exited"
#     input_cols = [c for c in raw_df.columns if c not in ("CustomerId", target_col)]
#
#     X = raw_df[input_cols].copy()
#     y = raw_df[target_col].astype(int)
#
#     # 4) Спліт: 60% train, 20% val, 20% test
#     X_train, X_temp, y_train, y_temp = train_test_split(
#         X, y, test_size=0.4, random_state=42, stratify=y
#     )
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
#     )
#
#     # 5) Виявляємо типи колонок
#     numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
#     categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
#
#     # 6) Імп'ютація числових
#     num_imputer = SimpleImputer(strategy="mean").fit(X_train[numeric_cols])
#     X_train[numeric_cols] = num_imputer.transform(X_train[numeric_cols])
#     X_val[numeric_cols] = num_imputer.transform(X_val[numeric_cols])
#     X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
#
#     # 7) Масштабування числових
#     scaler = MinMaxScaler().fit(X_train[numeric_cols])
#     X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
#     X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
#     X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
#
#     # 8) One-hot кодування категоріальних (зворотна сумісність для різних версій sklearn)
#     try:
#         encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X_train[categorical_cols])
#     except TypeError:
#         # Для старіших версій scikit-learn
#         encoder = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(X_train[categorical_cols])
#
#     encoded_col_names = list(encoder.get_feature_names_out(categorical_cols))
#
#     def encode_and_concat(df):
#         if categorical_cols:
#             encoded = encoder.transform(df[categorical_cols])
#             encoded_df = pd.DataFrame(encoded, index=df.index, columns=encoded_col_names)
#             out = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
#         else:
#             out = df.copy()
#         return out
#
#     X_train = encode_and_concat(X_train)
#     X_val = encode_and_concat(X_val)
#     X_test = encode_and_concat(X_test)
#
#     return {
#         "train_X": X_train,
#         "train_y": y_train,
#         "val_X": X_val,
#         "val_y": y_val,
#         "test_X": X_test,
#         "test_y": y_test,
#         "numeric_cols": numeric_cols,
#         "categorical_cols": categorical_cols,
#         "encoded_cols": encoded_col_names,
#     }

# process_bank_churn.py
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


def prepare_datasets_from_df(
    df: pd.DataFrame,
    target_col: str = "Exited",
    required_cols: List[str] = None,
    random_state: int = 42
) -> DatasetBundle:
    """
    Версія prepare_datasets, що приймає DataFrame замість імені файлу.
    Використовуйте це в Jupyter/Colab, коли ви вже завантажили дані.
    """
    if required_cols is None:
        required_cols = ["CustomerId", target_col]

    # Clean
    raw_df = drop_required_na(df, required_cols)

    # Features/target split
    X, y = split_features_target(raw_df, target_col=target_col, drop_cols=["CustomerId"])

    # Split to train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20(X, y, random_state=random_state)

    # Column types
    numeric_cols, categorical_cols = detect_column_types(X_train)

    # Numeric: impute → scale
    num_imputer = fit_numeric_imputer(X_train, numeric_cols)
    X_train = transform_numeric_imputer(num_imputer, X_train, numeric_cols)
    X_val   = transform_numeric_imputer(num_imputer, X_val,   numeric_cols)
    X_test  = transform_numeric_imputer(num_imputer, X_test,  numeric_cols)

    scaler = fit_scaler(X_train, numeric_cols)
    X_train = transform_scaler(scaler, X_train, numeric_cols)
    X_val   = transform_scaler(scaler, X_val,   numeric_cols)
    X_test  = transform_scaler(scaler, X_test,  numeric_cols)

    # Categorical: one-hot
    encoder = fit_onehot_encoder(X_train, categorical_cols)
    X_train, encoded_cols = transform_onehot_concat(encoder, X_train, categorical_cols)
    X_val,   _            = transform_onehot_concat(encoder, X_val,   categorical_cols)
    X_test,  _            = transform_onehot_concat(encoder, X_test,  categorical_cols)

    return DatasetBundle(
        train_X=X_train,
        train_y=y_train,
        val_X=X_val,
        val_y=y_val,
        test_X=X_test,
        test_y=y_test,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        encoded_cols=encoded_cols,
    )
    

# --------------------------- Data structures ---------------------------

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
        # Створюємо "порожній" енкодер; використовуватиметься умовна гілка під час трансформації
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


# --------------------------- Orchestration (new public API) ---------------------------

def prepare_datasets(
    csv_name: str = "train.csv",
    target_col: str = "Exited",
    required_cols: List[str] = None,
    random_state: int = 42
) -> DatasetBundle:
    """
    Основна публічна функція модуля.
    1) Зчитує дані з CSV (відносно модуля),
    2) чистить обов'язкові колонки,
    3) ділить на train/val/test (60/20/20),
    4) імп'ютить та масштабує числові,
    5) one-hot кодує категоріальні,
    6) повертає структурований контейнер DatasetBundle.
    """
    if required_cols is None:
        required_cols = ["CustomerId", target_col]

    # Read & clean
    data_path = resolve_data_path(csv_name)
    raw_df = read_csv_at(data_path, index_col=0)
    raw_df = drop_required_na(raw_df, required_cols)

    # Features/target split
    X, y = split_features_target(raw_df, target_col=target_col, drop_cols=["CustomerId"])

    # Split to train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20(X, y, random_state=random_state)

    # Column types
    numeric_cols, categorical_cols = detect_column_types(X_train)

    # Numeric: impute → scale
    num_imputer = fit_numeric_imputer(X_train, numeric_cols)
    X_train = transform_numeric_imputer(num_imputer, X_train, numeric_cols)
    X_val   = transform_numeric_imputer(num_imputer, X_val,   numeric_cols)
    X_test  = transform_numeric_imputer(num_imputer, X_test,  numeric_cols)

    scaler = fit_scaler(X_train, numeric_cols)
    X_train = transform_scaler(scaler, X_train, numeric_cols)
    X_val   = transform_scaler(scaler, X_val,   numeric_cols)
    X_test  = transform_scaler(scaler, X_test,  numeric_cols)

    # Categorical: one-hot
    encoder = fit_onehot_encoder(X_train, categorical_cols)
    X_train, encoded_cols = transform_onehot_concat(encoder, X_train, categorical_cols)
    X_val,   _            = transform_onehot_concat(encoder, X_val,   categorical_cols)
    X_test,  _            = transform_onehot_concat(encoder, X_test,  categorical_cols)

    return DatasetBundle(
        train_X=X_train,
        train_y=y_train,
        val_X=X_val,
        val_y=y_val,
        test_X=X_test,
        test_y=y_test,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        encoded_cols=encoded_cols,
    )


# --------------------------- Backward compatibility ---------------------------

def process_data(csv_name: str = "train.csv") -> Dict[str, pd.DataFrame | pd.Series | List[str]]:
    """
    [DEPRECATED] Історична точка входу.
    Використовуйте `prepare_datasets(...)`.
    Повертає той самий словник, що й раніше, для сумісності зі старим кодом.
    """
    warnings.warn(
        "`process_data(...)` застаріла. Використовуйте `prepare_datasets(...)`.",
        DeprecationWarning,
        stacklevel=2
    )
    bundle = prepare_datasets(csv_name=csv_name)

    return {
        "train_X": bundle.train_X,
        "train_y": bundle.train_y,
        "val_X": bundle.val_X,
        "val_y": bundle.val_y,
        "test_X": bundle.test_X,
        "test_y": bundle.test_y,
        "numeric_cols": bundle.numeric_cols,
        "categorical_cols": bundle.categorical_cols,
        "encoded_cols": bundle.encoded_cols,
    }
