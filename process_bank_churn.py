from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def process_data(csv_name: str = "train.csv"):
    """
    Читає дані, чистить, ділить на train/val/test, імп'ютить, масштабує числові
    та one-hot кодує категоріальні ознаки. Повертає словник з наборами даних.
    """
    # 1) Читаємо CSV відносно поточного файлу модуля (стабільніше за CWD)
    data_path = Path(__file__).resolve().parent / csv_name
    raw_df = pd.read_csv(data_path, index_col=0)

    # 2) Прибираємо рядки з пропусками у ключових колонках
    raw_df = raw_df.dropna(subset=["CustomerId", "Exited"]).copy()

    # 3) Визначаємо колонки ознак і таргет
    target_col = "Exited"
    input_cols = [c for c in raw_df.columns if c not in ("CustomerId", target_col)]

    X = raw_df[input_cols].copy()
    y = raw_df[target_col].astype(int)

    # 4) Спліт: 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 5) Виявляємо типи колонок
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # 6) Імп'ютація числових
    num_imputer = SimpleImputer(strategy="mean").fit(X_train[numeric_cols])
    X_train[numeric_cols] = num_imputer.transform(X_train[numeric_cols])
    X_val[numeric_cols] = num_imputer.transform(X_val[numeric_cols])
    X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    # 7) Масштабування числових
    scaler = MinMaxScaler().fit(X_train[numeric_cols])
    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # 8) One-hot кодування категоріальних (зворотна сумісність для різних версій sklearn)
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X_train[categorical_cols])
    except TypeError:
        # Для старіших версій scikit-learn
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(X_train[categorical_cols])

    encoded_col_names = list(encoder.get_feature_names_out(categorical_cols))

    def encode_and_concat(df):
        if categorical_cols:
            encoded = encoder.transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded, index=df.index, columns=encoded_col_names)
            out = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
        else:
            out = df.copy()
        return out

    X_train = encode_and_concat(X_train)
    X_val = encode_and_concat(X_val)
    X_test = encode_and_concat(X_test)

    return {
        "train_X": X_train,
        "train_y": y_train,
        "val_X": X_val,
        "val_y": y_val,
        "test_X": X_test,
        "test_y": y_test,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "encoded_cols": encoded_col_names,
    }
