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
                plot_every_n_epochs: int,
                to_save_plots: bool,
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

    def force_non_negative_exponents_(self):
        with torch.no_grad():
            self.l1.weight.clamp_(0.)


    def forward(self, x):
        # return self.l2(torch.exp(self.l1(torch.log(self.l0(x.view(x.size(0), -1))))))
        return self.l2(torch.exp(self.l1(torch.log(x.view(x.size(0), -1)))))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, nesterov=True)

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

    def on_train_batch_end(self, outputs, batch, batch_idx, unused = 0):
        self.force_non_negative_exponents_()
        return # gets called after optimizer_step()
 
    def on_train_start(self):
        # Track total time
        self.st_total = time.time()

        # Initialize list with tracked layerweights
        self.exponent_path = [self.get_exponents()]
        self.coefficient_path = [self.get_coefficients()]
        self.bias_path = [self.get_bias()]

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
        # exponents = self.get_exponents() #column vector [n, 1]
        # for i in range(len(exponents)):
        #     self.log(f"exponent/#{i+1}", exponents[i, :].item(), prog_bar=True)
        # coefficients = self.get_coefficients() #column vector [n, 1]
        # for i in range(len(coefficients)):
        #     self.log(f"coefficient/#{i+1}", coefficients[i, :].item(), prog_bar=True)
        # bias = self.get_bias() #column vector [n, 1]
        # for i in range(len(bias)):
        #     self.log(f"bias/#{i+1}", bias[i, :].item(), prog_bar=True)
        
        # Plot predictions, exponents and coefficients
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

            # Condense paths into arrays
            exponent_path = np.stack(self.exponent_path).squeeze(-1) #shape(2, 3)
            coefficient_path = np.stack(self.coefficient_path).squeeze(-1) # shape (2, 3)
            bias_path = np.stack(self.bias_path).squeeze(-1) # shape (2, 1)

            # if more than 3 hidden neurons, cant show necessary rank of weights in layer1
            if exponent_path.shape[-1] > 3:
                fig, ax = plt.subplots(1, 2, figsize = (14, 5))
                # Exponents
                for i in range(exponent_path.shape[-1]):
                    ax[0].plot(exponent_path[:, i])
                if self.hparams.target_fn.__name__ == "constantf":
                    ax[0].axhline(0, label='Target Rank', c="red", ls="--")
                if self.hparams.target_fn.__name__ == "linearf":
                    ax[0].axhline(1, label='Target Rank', c="red", ls="--")
                if self.hparams.target_fn.__name__ == "polynomialf":
                    ax[0].axhline(3, label='Target Rank', c="red", ls="--") #rank 3 monomial
                    ax[0].axhline(2, label='Target Rank', c="red", ls="--") #rank 2 monomial
                    ax[0].axhline(1, label='Target Rank', c="red", ls="--")      
                    ax[0].axhline(0, label='Target Rank', c="red", ls="--")
                
                ax[0].set_title("Learned Exponent Paths")
                ax[0].set_xlabel("Epoch")
                ax[0].set_ylabel("Exponent Value")
                ax[0].legend()

                # Coefficients
                for i in range(coefficient_path.shape[-1]):
                    ax[1].plot(coefficient_path[:, i])
                                 
                ax[1].set_title("Learned Coefficient Paths")
                ax[1].set_xlabel("Epoch")
                ax[1].set_ylabel("Coefficient Value")
                ax[1].legend()
            
            else:
                # Sort from lowest to largest exponent
                ind = np.argsort(exponent_path[-1])
                exponent_path = exponent_path[:, ind]
                coefficient_path = coefficient_path[:, ind]
                coefficient_path = np.hstack([bias_path, coefficient_path])

                # Prepare for plotting
                fig, ax = plt.subplots(1, 2, figsize = (14, 5))
                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = prop_cycle.by_key()['color']

                # coefficients
                coefficients = np.array([2, -15, 36, -25]) / 10

                # Plot the exponents path
                for i, path in enumerate(exponent_path.T):
                    label = "Target Exponents" if i == 0 else None
                    ax[0].plot(np.full(self.current_epoch+1, i+1), label=label, c=colors[i], ls="--")
                    label = "Learned exponents" if i == 0 else None
                    ax[0].plot(path, label=label, c=colors[i])

                plot_bottom = ax[0].get_ylim()[0]
                if plot_bottom > 0:
                    ax[0].set_ylim(0)
                        
                ax[0].set_title("Learned Exponent Paths")
                ax[0].set_xlabel("Epoch")
                ax[0].set_ylabel("Exponent Value")
                ax[0].legend()

                # Plot the coefficient paths
                for degree, (coefficient, path) in enumerate(zip(coefficients[::-1], coefficient_path.T)):
                    label = "Target coefficient" if degree == 0 else None
                    ax[1].plot(np.full(self.current_epoch+1, coefficient), label=label, c=colors[degree], ls="--")
                    label = f"Learned {degree}-th degree coefficient"
                    ax[1].plot(path, label=label, c=colors[degree])
                        
                ax[1].set_title("Learned Coefficient Paths")
                ax[1].set_xlabel("Epoch")
                ax[1].set_ylabel("Coefficient Value")
                ax[1].legend()

            if self.hparams.to_save_plots:
                # save plot in current logging directory
                path = os.path.join(self.logger.log_dir, "plots")
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, f"exponents_coefficients_{self.current_epoch}.png")
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
