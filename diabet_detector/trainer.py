import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAveragePrecision


class DiabetDetector(pl.LightningModule):
    """Module for training and evaluating model"""

    def __init__(self, model, class_weights=None, lr=1e-4):
        super().__init__()
        self.model = model
        self.class_weights = class_weights
        self.loss_fn = torch.nn.BCELoss()
        self.lr = lr
        self.metric = BinaryAveragePrecision()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        preds = self.model(data)
        loss = self.loss_fn(preds, labels.type(dtype=torch.float32))
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        preds = self.model(data)
        loss = self.loss_fn(preds, labels.type(dtype=torch.float32))
        ap_score = self.metric(preds, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_AP", ap_score, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_AP": ap_score}

    def test_step(self, batch, batch_idx):
        data, labels = batch
        preds = self.model(data)
        ap_score = self.metric(preds, labels)
        self.log("test_AP", ap_score, prog_bar=True, on_epoch=True)
        return {"test_AP": ap_score}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
