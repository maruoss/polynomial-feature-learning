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


class LinearModel(pl.LightningModule):
    def __init__(self, 
                input_dim: int, 
                hidden_dim: int, 
                learning_rate: float, 
                datamodule,
                low_oos: float,
                high_oos: float,
                scale: bool,
                oos: bool,
                target_fn,
                show_orig_scale: bool,
                plot_every_n_epochs: int,
                to_save_plots: bool,
                ):
        super().__init__()
        self.save_hyperparameters()

        # Layers
        self.l1 = nn.Linear(input_dim, 1, bias = True)
    

    def forward(self, x):
        return self.l1(x.view(x.size(0), -1))

    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.8, nesterov=True)

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1., norm_type=2.0, error_if_nonfinite=True)
        # torch.nn.utils.clip_grad_value_(self.parameters(), 1)
        return

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/loss", loss, on_epoch=False, prog_bar=True)
        # self.log('metrics/r2', r2_score(y_hat, y), on_epoch=False, prog_bar=True)

        return loss
    
    def on_train_start(self):
        self.st_total = time.time()
        self.plotter = predictions(datamodule=self.hparams.datamodule, model=self, low_oos=self.hparams.low_oos, 
                                    high_oos=self.hparams.high_oos, scale=self.hparams.scale, oos=self.hparams.oos, 
                                    target_fn=self.hparams.target_fn, show_orig_scale=self.hparams.show_orig_scale)

    def on_train_epoch_start(self):
        self.st = time.time()
        self.steps = self.global_step

    def on_train_epoch_end(self):
        elapsed = time.time() - self.st
        steps_done = self.global_step - self.steps
        self.log("time/step", elapsed / steps_done)

    def on_train_end(self):
        elapsed = time.time() - self.st_total
        print(f"Total Training Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/val_loss", loss, prog_bar=True)
        # self.log("metrics/val_r2", r2_score(y_hat, y), prog_bar=True)

        return {"val_loss": loss,"y_hat": y_hat}

    def on_validation_epoch_end(self):
        # Plot predictions
        if (self.current_epoch+1) % (self.hparams.plot_every_n_epochs) == 0: # +1 because 10th epoch is counted as 9 starting at 0
            # Plot predictions
 
            self.plotter.plot()
            
            if self.hparams.to_save_plots:
                # save plot in current logging directory
                path = os.path.join(self.logger.log_dir, "plots")
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, f"predictions_{self.current_epoch}.png")
                plt.savefig(path, facecolor="white")
                plt.close()

            plt.show() #to free memory
        return

    # def validation_epoch_end(self, outputs):
        # avg_pred = torch.cat([x["y_hat"] for x in outputs]).mean()
        # self.log("avg_pred_epoch", avg_pred, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/test_loss", loss, prog_bar=True)
        # self.log("metrics/test_r2", r2_score(y_hat, y), prog_bar=True)
        return loss
