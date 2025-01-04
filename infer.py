import hydra
import numpy as np
import pandas as pd
import torch
from dvc.api import DVCFileSystem
from omegaconf import DictConfig

from diabet_detector.data_preprocessing import preprocess_data
from diabet_detector.model import SimpleClassifier
from diabet_detector.trainer import DiabetDetector


def inference(data: pd.DataFrame, model_path: str, scaler_path: str) -> np.ndarray:
    data_tensor = preprocess_data(data=data, path_to_scaler=scaler_path)
    fs = DVCFileSystem()
    with fs.open(model_path) as model_data:
        model = DiabetDetector.load_from_checkpoint(
            model_data, model=SimpleClassifier()
        )
    model.eval()
    with torch.no_grad():
        preds = model(data_tensor.to(model.device))
    return preds.numpy(force=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    df = pd.read_csv(config["inference"]["data_path"])
    preds = inference(
        data=df,
        model_path=config["inference"]["model_path"],
        scaler_path=config["data_loading"]["scaler_path"],
    )
    print(preds)


if __name__ == "__main__":
    main()
