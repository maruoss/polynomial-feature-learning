import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class PolyModel(pl.LightningModule):
    def __init__(self, 
                input_dim: int, 
                hidden_dim: int, 
                learning_rate: float,
                ):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, 1, bias=True)

        with torch.no_grad(): # should be preferred instead of weight.data.uniform
            self.l1.weight.uniform_(0., 2./ input_dim) # Dont initialize negative values. To mitigate nan problem.
    
    def get_exponents(self):
        return self.l1.weight.detach().cpu().numpy()
    
    def get_coefficients(self):
        return self.l2.weight.detach().cpu().numpy().reshape(-1, 1) #reshape as its a row vector
    
    def get_bias(self):
        return self.l2.bias.detach().cpu().numpy().reshape(-1, 1)

    def force_non_negative_exponents_(self):
        with torch.no_grad():
            self.l1.weight.clamp_(0.)

    def forward(self, x):
        return self.l2(torch.exp(self.l1(torch.log(x.view(x.size(0), -1)))))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, nesterov=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/loss", loss, on_epoch=False, prog_bar=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, unused = 0):
        self.force_non_negative_exponents_()
        return # gets called after optimizer_step()

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
