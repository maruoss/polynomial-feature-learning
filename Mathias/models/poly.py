import time
import numpy as np
import torch
from torch import nn
from torchmetrics.functional import r2_score
from torch.nn import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from plotter import predictions


class PolyModel(pl.LightningModule):
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
                to_plot: bool,
                # plot_every: bool,
                ):
        super().__init__()
        self.save_hyperparameters()

        # self.l0 = nn.Linear(input_dim, input_dim, bias=False) # even when not used influences random initialization of l1 and l2 layers (random process)
        self.l1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, 1, bias=True)

        with torch.no_grad(): # should be preferred instead of weight.data.uniform
            # Layer 0: dont initialize negative weights!
            # self.l0.weight.uniform_(0., 1. / np.sqrt(input_dim))
            # Layer1: Weights = Exponents (shouldnt be negative!)
            self.l1.weight.uniform_(0., 1.) # Dont initialize negative values. To mitigate nan problem.
            # Layer2: Weigths = Coefficients (Linear comb. of monomials)
            # self.l2.weight.uniform_(-1. / np.sqrt(input_dim), 1. / np.sqrt(input_dim)) # Dont initialize negative values -> can get l1 weight to negative weights!
            self.l2.weight.uniform_(0., 1. / np.sqrt(input_dim)) # Dont initialize negative values -> can get l1 weight to negative weights!
    
    def get_exponents(self):
        return self.l1.weight.detach().cpu().numpy()
    
    def get_coefficients(self):
        return self.l2.weight.detach().cpu().numpy().reshape(-1, 1) #reshape as its a row vector
    
    def get_bias(self):
        return self.l2.bias.detach().cpu().numpy().reshape(-1, 1)

    def forward(self, x):
        # return self.l2(torch.exp(self.l1(torch.log(self.l0(x.view(x.size(0), -1))))))
        return self.l2(torch.exp(self.l1(torch.log(x.view(x.size(0), -1)))))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.8, nesterov=True)

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        # # Clip gradients of exponents and coefficients together
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1., norm_type=2.0, error_if_nonfinite=True)
        # torch.nn.utils.clip_grad_value_(self.parameters(), 1)

        # # Clip gradients separately
        # for name, param in self.named_parameters():
        #     if name in ['l1.weight']:
        #         #1: Value Clipping
        #         # nn.utils.clip_grad_value_(param, 1)
        #         # #2:  Norm clipping
        #         torch.nn.utils.clip_grad_norm_(param, 1., norm_type=2., error_if_nonfinite=True)
        #         # # 3: Round gradients of exponents to be integers
        #         # param.grad.data.round_() # round so that update to exponents are constrained to be integers
        #     if name in ['l2.weight']:
        #         torch.nn.utils.clip_grad_norm_(param, 1., norm_type=2.0, error_if_nonfinite=True)
        #         pass
        return

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/loss", loss, on_epoch=False, prog_bar=True)
        # self.log('metrics/r2', r2_score(y_hat, y), on_epoch=False, prog_bar=True)

        return loss

    def on_train_start(self):
        # Track total time
        self.st_total = time.time()

        # Initialize list with tracked layerweights
        self.exponent_path = [self.get_exponents()]
        self.coefficient_path = [self.get_coefficients()]
        self.bias_path = [self.get_bias()]

    def on_train_epoch_start(self):
        self.st = time.time()
        self.steps = self.global_step

    def on_train_epoch_end(self):
        elapsed = time.time() - self.st
        steps_done = self.global_step - self.steps
        self.log("time/step", elapsed / steps_done)

        # Append exponents, coeffcients, bias to respective lists
        self.exponent_path.append(self.get_exponents())
        self.coefficient_path.append(self.get_coefficients())
        self.bias_path.append(self.get_bias())

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
        # Log exponents, coefficients and biases
        exponents = self.get_exponents()
        for i in range(len(exponents)):
            self.log(f"exponent.#{i+1}", exponents[i, :].item(), prog_bar=True)
        coefficients = self.get_coefficients()
        for i in range(len(coefficients)):
            self.log(f"coefficient.#{i+1}", coefficients[i, :].item(), prog_bar=True)
        bias = self.get_bias()
        for i in range(len(bias)):
            self.log(f"bias.#{i+1}", bias[i, :].item(), prog_bar=True)
        
        # Plot predictions, exponents and coefficients
        if self.hparams.to_plot:
            # if self.current_epoch % self.hparams.plot_every == 0:
            # Plot predictions each epoch
            plotter = predictions(datamodule=self.hparams.datamodule, model=self, low_oos=self.hparams.low_oos, 
                                        high_oos=self.hparams.high_oos, scale=self.hparams.scale, oos=self.hparams.oos, 
                                        target_fn=self.hparams.target_fn, show_orig_scale=self.hparams.show_orig_scale) 
            plotter.plot()
            
            # Exponents
            self.templ1weights = np.stack(self.exponent_path).squeeze(axis=-1) #only last axis if dim 0 is also 1
            for i in range(self.templ1weights.shape[-1]):
                plt.plot(self.templ1weights[:, i])
                if self.hparams.target_fn.__name__ == "constantf":
                    plt.axhline(0, label='Target Rank', c="red", ls="--")
                if self.hparams.target_fn.__name__ == "linearf":
                    plt.axhline(1, label='Target Rank', c="red", ls="--")
                if self.hparams.target_fn.__name__ == "polynomialf":
                    plt.axhline(3, label='Target Rank', c="red", ls="--") #rank 3 monomial
                    plt.axhline(2, label='Target Rank', c="red", ls="--") #rank 2 monomial
                    plt.axhline(1, label='Target Rank', c="red", ls="--")      
                    plt.axhline(0, label='Target Rank', c="red", ls="--")
            plt.show()

            # Coefficients
            self.templ2weights = np.stack(self.coefficient_path).squeeze(axis=-1)
            for i in range(self.templ2weights.shape[-1]):
                plt.plot(self.templ2weights[:, i])
                # if self.hparams.target_fn.__name__ == "constantf":
                #     plt.axhline(0, label='Target Rank', c="red", ls="--")
                # if self.hparams.target_fn.__name__ == "linearf":
                #     plt.axhline(1, label='Target Rank', c="red", ls="--")
                # if self.hparams.target_fn.__name__ == "polynomialf":
                #     plt.axhline(3, label='Target Rank', c="red", ls="--") #rank 3 monomial
                #     plt.axhline(2, label='Target Rank', c="red", ls="--") #rank 2 monomial
                #     plt.axhline(1, label='Target Rank', c="red", ls="--")      
                #     plt.axhline(0, label='Target Rank', c="red", ls="--")
            plt.show()


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
