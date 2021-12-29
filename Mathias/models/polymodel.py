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
                mode: str,
                datamodule,
                low_oos: float,
                high_oos: float,
                scale: bool,
                oos: bool,
                target_fn,
                show_orig_scale: bool,
                to_plot: bool,
                plot_every: bool,
                ):
        super().__init__()
        self.save_hyperparameters()
        # Poly and Standard NN regression layers
        if mode in ["standard", "poly"]:
            # self.l0 = nn.Linear(input_dim, input_dim, bias=False) # even when not used influences random initialization of l1 and l2 layers (random process)
            self.l1 = nn.Linear(input_dim, hidden_dim, bias=False)
            self.l2 = nn.Linear(hidden_dim, 1, bias=True)

        if mode == "poly":
            # self.l0 = nn.Linear(input_dim, input_dim, bias=False)
            with torch.no_grad(): # should be preferred instead of weight.data.uniform
                # Layer 0: dont initialize negative weights!
                # self.l0.weight.uniform_(0., 1. / np.sqrt(input_dim))
                # Layer1: Weights = Exponents (shouldnt be negative!)
                self.l1.weight.uniform_(0., 1.) # Dont initialize negative values. To mitigate nan problem.
                # Layer2: if mode=="poly": Weigths = Coefficients (Linear comb. of monomials)
                # self.l2.weight.uniform_(-1. / np.sqrt(input_dim), 1. / np.sqrt(input_dim)) # Dont initialize negative values -> can get l1 weight to negative weights!
                self.l2.weight.uniform_(0., 1. / np.sqrt(input_dim)) # Dont initialize negative values -> can get l1 weight to negative weights!

        # Linear regression
        if mode == "linear":
            self.l3 = nn.Linear(input_dim, 1, bias = True)

    def forward(self, x):
        # Option 1: Poly Feature Learning
        if self.hparams.mode == "poly":

            # Uncomment/ Comment out here whether to use layer 0 or not:
            # return self.l2(torch.exp(self.l1(torch.log(self.l0(x.view(x.size(0), -1))))))
            return self.l2(torch.exp(self.l1(torch.log(x.view(x.size(0), -1)))))
        # Option 2: Standard 1-hidden layer MLP
        elif self.hparams.mode == "standard":
            return self.l2(torch.sigmoid(self.l1(x.view(x.size(0), -1))))
        # Option 3: Linear regression
        elif self.hparams.mode == "linear":
            return self.l3(x.view(x.size(0), -1))
        else:
            raise NotImplementedError("Please specify a mode of either 'poly', 'standard' or 'linear'")
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.8, nesterov=True)

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1., norm_type=2.0, error_if_nonfinite=True)
        # if self.hparams.mode == "poly":
            # nn.utils.clip_grad_value_(self.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1., norm_type=2.0, error_if_nonfinite=True)
            # breakpoint()
            # for name, param in self.named_parameters():
                # if name in ['l1.weight']:
            #         # #1: Value Clipping
            #         # nn.utils.clip_grad_value_(param, 1)
            #         # #2:  Norm clipping
                    # torch.nn.utils.clip_grad_norm_(param, 1., norm_type=2., error_if_nonfinite=True)
                    # breakpoint()
                    # pass
                    # # 3: Round gradients of exponents to be integers
                    # param.grad.data.round_() # round so that update to exponents are constrained to be integers
                # if name in ['l2.weight']:
                    # torch.nn.utils.clip_grad_norm_(param, 1., norm_type=2.0, error_if_nonfinite=True)
                    # pass
            return

    def training_step(self, batch, batch_idx):
        x, y = batch
        # breakpoint()
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/loss", loss, on_epoch=False, prog_bar=True)
        # self.log('metrics/r2', r2_score(y_hat, y), on_epoch=False, prog_bar=True)

        return loss
    
    def on_train_start(self):
        # Track exponents and coefficients
        self.layer1weights = []
        self.layer2weights = []
        self.st_total = time.time()

    def on_train_epoch_start(self):
        self.st = time.time()
        self.steps = self.global_step

    def training_epoch_end(self, outputs) -> None:

        for name, params in self.named_parameters():
            # # Gives error when weights are nan?
            # self.logger.experiment.add_histogram(name, params, self.current_epoch)
            if name in ["l1.weight"]:
                for i in range(len(self.l1.weight)): #column vector
                    self.log(f"weights/ l1.weight.#{i+1}", self.l1.weight.data.clone()[i, :], prog_bar=True)
                    self.layer1weights.append(torch.cat([params.data.clone(), torch.tensor(self.global_step, device=self.device).view(-1, 1)])) #clone to not get pointer only
            if name in ["l2.weight"]:
                for i in range(self.l2.weight.shape[-1]): #row vector
                    self.log(f"weights/ l2.weight.#{i+1}", self.l2.weight.data.clone()[:, i], prog_bar=True) #l2 weight is row vector
                    self.layer2weights.append(torch.cat([params.data.clone().view(-1, 1), torch.tensor(self.global_step, device=self.device).view(-1, 1)])) # l2weights are row vectors in contrast to l1 (column)


    def on_train_epoch_end(self):
        elapsed = time.time() - self.st
        steps_done = self.global_step - self.steps
        self.log("time/step", elapsed / steps_done)

        if self.hparams.to_plot:
            if self.current_epoch % self.hparams.plot_every == 0:
                # Plot predictions each epoch
                plotter = predictions(datamodule=self.hparams.datamodule, model=self, low_oos=self.hparams.low_oos, 
                                            high_oos=self.hparams.high_oos, scale=self.hparams.scale, oos=self.hparams.oos, 
                                            target_fn=self.hparams.target_fn, show_orig_scale=self.hparams.show_orig_scale) 
                plotter.plot()
                
                # layer1 weights
                if self.layer1weights: #if list not empty (as in linear regression with no l1 weights, but only l3)
                    self.templ1weights = torch.stack(self.layer1weights, dim=0).squeeze()
                    for i in range(self.templ1weights.shape[-1]-1): # minus index (training step) column in last column
                        ind = self.templ1weights.shape[-1] - 1
                        plt.plot(self.templ1weights[:, ind].cpu(), self.templ1weights[:, i].cpu())
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
                
                if self.layer2weights: #if list not empty (as in linear regression with no l1 weights, but only l3)
                    self.templ2weights = torch.stack(self.layer2weights, dim=0).squeeze()
                    for i in range(self.templ2weights.shape[-1]-1): # minus index (training step) column in last column
                        ind = self.templ2weights.shape[-1] - 1
                        plt.plot(self.templ2weights[:, ind].cpu(), self.templ2weights[:, i].cpu())
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

    def on_train_end(self):
        if self.layer1weights: #if list not empty (as in linear regression with no l1 weights, but only l3)
            self.layer1weights = torch.stack(self.layer1weights, dim=0).squeeze()
        
        if self.layer2weights:
            self.layer2weights = torch.stack(self.layer2weights, dim=0).squeeze()

        elapsed = time.time() - self.st_total
        print(f"Total Training Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/val_loss", loss, prog_bar=True)
        # self.log("metrics/val_r2", r2_score(y_hat, y), prog_bar=True)

        return {"val_loss": loss,"y_hat": y_hat}

    def validation_epoch_end(self, outputs):
        avg_pred = torch.cat([x["y_hat"] for x in outputs]).mean()
        self.log("avg_pred_epoch", avg_pred, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/test_loss", loss, prog_bar=True)
        # self.log("metrics/test_r2", r2_score(y_hat, y), prog_bar=True)
        return loss
