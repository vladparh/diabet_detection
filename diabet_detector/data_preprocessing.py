from typing import Tuple

import joblib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset


def prepare_data_for_training(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 123,
    path_to_save_scaler: str = None,
) -> Tuple[TensorDataset, TensorDataset]:
    """Function for prepare data from csv-file. Returns datasets for training and validation"""
    df = pd.read_csv(data_path)
    df.drop_duplicates(inplace=True)
    train_data, test_data, train_labels, test_labels = train_test_split(
        df.drop(columns=["Diabetes_binary"]),
        df["Diabetes_binary"],
        test_size=test_size,
        random_state=random_state,
    )
    scaler = MinMaxScaler()
    train_data[
        ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
    ] = scaler.fit_transform(
        train_data[
            ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
        ]
    )
    test_data[
        ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
    ] = scaler.transform(
        test_data[
            ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
        ]
    )

    if path_to_save_scaler is not None:
        joblib.dump(scaler, filename=path_to_save_scaler)

    train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.long)

    train_dataset = TensorDataset(train_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_tensor, test_labels_tensor)

    return train_dataset, test_dataset


def preprocess_data(data: pd.DataFrame, path_to_scaler: str) -> torch.Tensor:
    """Modify data from pandas DataFrame to torch Tensor"""
    scaler = joblib.load(path_to_scaler)
    data[
        ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
    ] = scaler.transform(
        data[["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]]
    )
    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    return data_tensor
