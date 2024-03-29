import time
import numpy as np
import torch
from torch import nn
from torchmetrics.functional import r2_score
from torch.nn import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os

from plotter import predictions


class StandardDeepModel(pl.LightningModule):
    def __init__(self, 
                input_dim: int, 
                hidden_dim: int, 
                learning_rate: float,
                datamodule,
                low_oos: float,
                high_oos: float,
                target_fn,
                plot_every_n_epochs: int,
                to_save_plots: bool,
                ):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l4 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l5 = nn.Linear(hidden_dim, 1, bias=True)

    def get_l1weights(self):
        return self.l1.weight.detach().cpu().numpy()
    
    def get_l2weights(self):
        return self.l2.weight.detach().cpu().numpy().reshape(-1, 1) #reshape as its a row vector
    
    def get_bias(self):
        return self.l2.bias.detach().cpu().numpy().reshape(-1, 1)

    def forward(self, x):
        return self.l5(torch.relu(self.l4(torch.relu(self.l3(torch.relu(self.l2(torch.relu(self.l1(x.view(x.size(0), -1))))))))))

  
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, nesterov=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/loss", loss, on_epoch=False, prog_bar=True)

        return loss
    
    def on_train_start(self):
        self.st_total = time.time()
        
        # # Initialize list with tracked layerweights
        # self.l1_weights = [self.get_l1weights()]
        # self.l2_weights = [self.get_l2weights()]
        # self.bias_path = [self.get_bias()]

    def on_train_epoch_start(self):
        self.st = time.time()
        self.steps = self.global_step

    def on_train_epoch_end(self):
        elapsed = time.time() - self.st
        steps_done = self.global_step - self.steps
        self.log("time/step", elapsed / steps_done)
    
        # # Append layerweights
        # self.l1_weights.append(self.get_l1weights())
        # self.l2_weights.append(self.get_l2weights())
        # self.bias_path.append(self.get_bias())

    def on_train_end(self):
        elapsed = time.time() - self.st_total
        print(f"Total Training Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/val_loss", loss, prog_bar=True)

        return {"val_loss": loss,"y_hat": y_hat}

    def on_validation_epoch_end(self):
        # Plot predictions
        if (self.current_epoch+1) % (self.hparams.plot_every_n_epochs) == 0: # +1 because 10th epoch is counted as 9 starting at 0
            # Plot predictions
            # Initialize plotter
            self.plotter = predictions(datamodule=self.hparams.datamodule, model=self, low_oos=self.hparams.low_oos, 
                            high_oos=self.hparams.high_oos, target_fn=self.hparams.target_fn) 

            self.plotter.plot()
            
            if self.hparams.to_save_plots:
                # save plot in current logging directory
                path = os.path.join(self.logger.log_dir, "plots")
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, f"predictions_{self.current_epoch}.png")
                plt.savefig(path, facecolor="white")
                plt.close()

            # plt.show() #to free memory

        return

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/test_loss", loss, prog_bar=True)
        return loss
