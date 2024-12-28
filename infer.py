import fire
import numpy as np
import pandas as pd
import torch

from diabet_detector.data_preprocessing import preprocess_data
from diabet_detector.model import SimpleClassifier
from diabet_detector.trainer import DiabetDetector


def inference(
    data_path: str, model_path: str, scaler_path: str = "./saved_models/scaler.pkl"
) -> np.ndarray:
    df = pd.read_csv(data_path)
    data_tensor = preprocess_data(data=df, path_to_scaler=scaler_path)
    model = DiabetDetector.load_from_checkpoint(
        model_path, model=SimpleClassifier(p_dropout=0.0)
    )
    model.eval()
    with torch.no_grad():
        preds = model(data_tensor.to(model.device))
    return preds.numpy(force=True)


if __name__ == "__main__":
    fire.Fire(inference)
