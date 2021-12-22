# %%
import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from argparse import ArgumentParser
from torchmetrics.functional import r2_score
from torch import nn

from numpy.random import default_rng
import pdb


# %%
# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(42, workers=True)

# Ignore warning in training when num_workers=0 (possible bottleneck)
import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# %%
# constant function for y values
def constantf(x: torch.Tensor, y: torch.FloatTensor) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs constant y
    in shape [obs, 1].
    """
    nobs = x.size(dim=0)
    y = torch.ones(nobs).reshape(-1, 1) * y
    return y


# %%
# create x values
def createargs(n, d, low, high) -> torch.Tensor():
    """
    Create n values of dimension d uniformly between [low, high]
    Input: n - number of observations
           d - dimension of observation
           low/ high - lower/ upper bounds of uniform distr.
    Output: [n, d] torch.FloatTensor
    """
    datamatrix = torch.FloatTensor(n, d).uniform_(low, high)
    return datamatrix

# %% Unit test
# Create dataset and groundtruths
# X = createargs(n=3, d=1, low=0, high=10)
# y = constantf(X)

# train_test_split(X, y, random_state=42,
#                 train_size=0.7, shuffle=True)


# %%
# DataModule
class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, n: int, d: int, low: float, high: float):
        """
        Args:
        n = number of observations to generate
        d = dimensions to generate
        low = lower bound of continuous uniform sampling
        high = upper bound of continous uniform sampling
        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        # Create full dataset
        self.X = createargs(n, d, low, high)
        self.y = constantf(self.X, y=3.)

    def setup(self, stage):
        # Split dataset into X, X_test
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y,
                train_size=0.8, shuffle=True)
        # Split X into X_train, X_val
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y,
                train_size=0.8, shuffle=True)
        
        print(f'# Training Examples: {len(self.y_train)}')
        print(f"Smallest value in train set: {torch.min(self.X_train)}")
        print(f"Biggest value in train set: {torch.max(self.X_train)}")
        print(f'# Validation Examples: {len(self.y_val)}')
        print(f'# Test Examples: {len(self.y_test)}')
        

    def example(self):
        """Returns a random training example."""        
        idx = np.random.randint(0, len(self.X_train))
        x, y = self.X_train[idx], self.y_train[idx]
        return (x, y)

    # return the dataloader for each split
    def train_dataloader(self):
        dataset = TensorDataset(self.X_train, self.y_train)
        return DataLoader(dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(dataset, batch_size=self.batch_size)

# %%
# Unit test randomness in splits
# a = MyDataModule()
# a.setup()
# a.example()


# %%
# Model
class PolyModel(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, learning_rate: float, mode: str="poly"):
        super().__init__()
        self.save_hyperparameters()
        # Polynomial Feature Learning
        # Layer1: Weights = Exponents (shouldnt be negative!)
        self.l1 = nn.Linear(input_dim, hidden_dim, bias=False)
        if mode == "poly":
            self.l1.weight.data.uniform_(0, 1. / np.sqrt(input_dim)) # Dont initialize negative values. To mitigate nan problem.
        # Layer2: if mode=="poly": Weigths = Coefficients (Linear comb. of monomials)
        self.l2 = nn.Linear(hidden_dim, 1, bias=False)
        if mode == "poly":
            self.l2.weight.data.uniform_(0, 1. / np.sqrt(input_dim)) # Dont initialize negative values.

    def forward(self, x):
        # # Option 1: Poly Feature Learning
        if self.hparams.mode == "poly":
            return self.l2(torch.exp(self.l1(torch.log(x.view(x.size(0), -1)))))
        # # Option 2: Standard 1-hidden layer MLP
        elif self.hparams.mode == "standard":
            return self.l2(F.sigmoid(self.l1(x.view(x.size(0), -1))))
        else:
            raise NotImplementedError
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        # nn.utils.clip_grad_value_(self.parameters(), 1)
        return

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        # breakpoint
        self.log("loss/loss", loss, on_epoch=False, prog_bar=True)
        self.log('metrics/r2', r2_score(y_hat, y), on_epoch=False, prog_bar=True)

        return loss
    
    def training_epoch_end(self, outputs) -> None:
        # tensorboard graph
        if (self.current_epoch==1):
            sample = torch.rand((1, self.hparams.input_dim))
            self.logger.experiment.add_graph(PolyModel(self.hparams.input_dim,
                self.hparams.hidden_dim, self.hparams.learning_rate), sample)
        
        # tensorboard weight histograms
        for name, params in self.named_parameters():
            # # Gives error when weights are nan?
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
            # breakpoint()
            self.log(f'mean {name}', params, prog_bar=True)

        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/val_loss", loss, prog_bar=True)
        self.log("metrics/val_r2", r2_score(y_hat, y), prog_bar=True)

        return {"val_loss": loss,"y_hat": y_hat}

    def validation_epoch_end(self, outputs):
        # breakpoint()
        avg_pred = torch.stack([x["y_hat"] for x in outputs]).mean()
        self.log("avg_pred_epoch", avg_pred, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/test_loss", loss, prog_bar=True)
        self.log("metrics/test_r2", r2_score(y_hat, y), prog_bar=True)
        return loss


# %%
# Hyperparameters
# Synthetic data X of shape [NUM_OBS, DIM] with values unif. cont. between LOW and HIGH
NUM_OBS = 10000
DIM = 1
LOW = 0.
HIGH = 10000.
# Function to learn #TODO: implement more than constant f.
TO_LEARN = "constant"

# Layer architecture
HIDDEN_DIM = 1      # equal to number of monomials if log/exp are used as activ. f.
MODE = "poly" # "poly": uses log+exp activations or "standard": uses sigmoid activation.
# output_dim = 1

# Learning algo
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# *********************
dm = MyDataModule(
    batch_size=BATCH_SIZE,
    n=NUM_OBS,
    d=DIM,
    low=LOW,
    high=HIGH,
)

model = PolyModel(
    input_dim=DIM,
    hidden_dim=HIDDEN_DIM,
    learning_rate=LEARNING_RATE,
    mode=MODE,
)

logger = pl.loggers.TensorBoardLogger(
    'logs', 
    name=f'mode-{MODE}.functolearn-{TO_LEARN}.low-{int(LOW)}.high-{int(HIGH)}.'\
    f'nobs-{NUM_OBS}.dim-{DIM}.lrate-{LEARNING_RATE}',
    default_hp_metric=False,

)
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    # default_root_dir='ckpts',
    gpus=1,
    logger=logger,
    log_every_n_steps=100,
    track_grad_norm=2,
)
trainer.fit(model, dm)
# %%
# trainer.test(
#     model, dm
# )

# %%




# %%
# def main(args):
#     model = PolyModel()
#     trainer = Trainer.from_argparse_args(args)
#     trainer.fit(model)


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser = Trainer.add_argparse_args(parser)
#     args = parser.parse_args()

#     main(args)




# %% Useful commands to debug
# print(list(self.parameters()))
