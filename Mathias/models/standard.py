import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class StandardModel(pl.LightningModule):
    def __init__(self, 
                input_dim: int, 
                hidden_dim: int, 
                learning_rate: float,
                ):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, 1, bias=True)

    def get_l1weights(self):
        return self.l1.weight.detach().cpu().numpy()
    
    def get_l2weights(self):
        return self.l2.weight.detach().cpu().numpy().reshape(-1, 1) #reshape as its a row vector
    
    def get_bias(self):
        return self.l2.bias.detach().cpu().numpy().reshape(-1, 1)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x.view(x.size(0), -1))))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, nesterov=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/loss", loss, on_epoch=False, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/val_loss", loss, prog_bar=True)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/test_loss", loss, prog_bar=True)
        return loss
