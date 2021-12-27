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
from sklearn.preprocessing import MinMaxScaler

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

# linear function
def linearf(x: torch.Tensor, slope:float=1., bias: float=0.) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs y
    in shape [obs, 1].
    """
    y = x * slope
    y = y.sum(axis=1, keepdim=True) + bias # sum across dimension and add single bias
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
# X = createargs(n=3, d=2, low=0, high=10)
# print(X)
# # y = constantf(X, y=3.)
# y = linearf(X, 2, 10)
# y
# %%
# y = y.sum(dim=1, keepdim=True) + 1.
# train_test_split(X, y, random_state=42,
#                 train_size=0.7, shuffle=True)

# %%
# scaler = MinMaxScaler(feature_range=(1e-6, 1))
# print(type(X))
# print(X)
# X = scaler.fit_transform(X)
# type(X)
# type(torch.from_numpy(X))

# %%
# DataModule
class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, n: int, d: int, low: float, high: float, funct:str, scale:bool):
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
        self.scale = scale
        # Create full dataset
        self.X = createargs(n, d, low, high)
        if funct == "constantf":
            self.y = constantf(self.X, y=3.) # set constant function value here
        elif funct == "linearf":
            self.y = linearf(self.X)
        else:
            raise NotImplementedError("Please specify a function via a string: 'constantf', 'linearf'")

    def setup(self, stage):
        # Split dataset into X, X_test
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y,
                train_size=0.8, shuffle=True)
        # Split X into X_train, X_val
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y,
                train_size=0.8, shuffle=True)
        
        if self.scale:
            scaler = MinMaxScaler(feature_range=(0.1, 1)) # SET SCALE RANGE -> 0 IS NOT SUITABLE FOR LN transform -> -inf
            assert self.X_train.dtype == self.X_val.dtype == self.X_test.dtype
            datatype = self.X_train.dtype
            # Scale train, val, testset
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_val = scaler.transform(self.X_val)
            self.X_test = scaler.transform(self.X_test)
            self.X_train = torch.from_numpy(self.X_train).to(datatype)
            self.X_val = torch.from_numpy(self.X_val).to(datatype)
            self.X_test = torch.from_numpy(self.X_test).to(datatype)

        
        print(f'# Training Examples: {len(self.y_train)} with X_train of shape {list(self.X_train.shape)}')
        print(f"Smallest value in train set: {torch.min(self.X_train)}")
        print(f"Biggest value in train set: {torch.max(self.X_train)}")
        print(f'# Validation Examples: {len(self.y_val)} with X_val of shape {list(self.X_val.shape)}')
        print(f'# Test Examples: {len(self.y_test)} with X_test of shape {list(self.X_test.shape)}')
        
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
# a = MyDataModule(batch_size=8, n=100, d=1, low=0., high=10., funct="constantf", scale=True)
# a.setup(stage=None)
# a.example()


# %%
# Model
class PolyModel(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, learning_rate: float, mode: str="poly"):
        super().__init__()
        self.save_hyperparameters()
        # Poly and Standard NN regression layers
        if mode in ["standard", "poly"]:
            # self.l0 = nn.Linear(input_dim, input_dim, bias=False) # even when not used influences random initialization of l1 and l2 layers (random process)
            self.l1 = nn.Linear(input_dim, hidden_dim, bias=False)
            self.l2 = nn.Linear(hidden_dim, 1, bias=False)

        if mode == "poly":
            # self.l0 = nn.Linear(input_dim, input_dim, bias=False)
            with torch.no_grad(): # should be preferred instead of weight.data.uniform
                # Layer 0: dont initialize negative weights!
                # self.l0.weight.uniform_(0., 1. / np.sqrt(input_dim))
                # Layer1: Weights = Exponents (shouldnt be negative!)
                self.l1.weight.uniform_(0., 0.) # Dont initialize negative values. To mitigate nan problem.
                # Layer2: if mode=="poly": Weigths = Coefficients (Linear comb. of monomials)
                self.l2.weight.uniform_(-1. / np.sqrt(input_dim), 1. / np.sqrt(input_dim)) # Dont initialize negative values

        # Linear regression
        if mode == "linear":
            self.l3 = nn.Linear(input_dim, 1, bias = True)

    def forward(self, x):
        # Option 1: Poly Feature Learning
        if self.hparams.mode == "poly":
            # with torch.no_grad(): # do not track these weight changes
            #     for name, param in self.named_parameters():
            #             if name in ['l1.weight']:
            #                 # below not necessary!: seems that gradient clip norm 1. and l1 weight init. to (0.,0.) is enough!
            #                 # F.relu_(param) # dont allow negative exponents
            #                 # param.round_() # only allow integer exponents
            #                 pass

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
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        if self.hparams.mode == "poly":
            # nn.utils.clip_grad_value_(self.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1., norm_type=2.0, error_if_nonfinite=True)
            # for name, param in self.named_parameters():
                # if name in ['l1.weight']:
            #         # #1: Value Clipping
            #         # nn.utils.clip_grad_value_(param, 1)
            #         # #2:  Norm clipping
                    # torch.nn.utils.clip_grad_norm_(param, 1., norm_type=2.0, error_if_nonfinite=True)
                    # pass
                    # # 3: Round gradients of exponents to be integers
                    # param.grad.data.round_() # round so that update to exponents are constrained to be integers
                # if name in ['l2.weight']:
                    # torch.nn.utils.clip_grad_norm_(param, 1., norm_type=2.0, error_if_nonfinite=True)
            return

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/loss", loss, on_epoch=False, prog_bar=True)
        self.log('metrics/r2', r2_score(y_hat, y), on_epoch=False, prog_bar=True)

        return loss
    
    def on_train_epoch_start(self):
        self.st = time.time()
        self.steps = self.global_step

    def training_epoch_end(self, outputs) -> None:
        # tensorboard graph
        # if (self.current_epoch==1):
        #     sample = torch.rand((1, self.hparams.input_dim))
        #     self.logger.experiment.add_graph(PolyModel(self.hparams.input_dim,
        #         self.hparams.hidden_dim, self.hparams.learning_rate), sample)
        
        # tensorboard weight histograms
        for name, params in self.named_parameters():
            # # Gives error when weights are nan?
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
            # breakpoint()
            self.log(f'weights/mean {name}', params, prog_bar=True)

        return

    def on_train_epoch_end(self):
        elapsed = time.time() - self.st
        steps_done = self.global_step - self.steps
        self.log("time/step", elapsed / steps_done)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/val_loss", loss, prog_bar=True)
        self.log("metrics/val_r2", r2_score(y_hat, y), prog_bar=True)

        return {"val_loss": loss,"y_hat": y_hat}

    def validation_epoch_end(self, outputs):
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
HIGH = 1000.

# Function to learn ("constantf", "linearf")
TO_LEARN = "constantf"
SCALE = False

# Layer architecture
HIDDEN_DIM = 1      # equal to number of monomials if log/exp are used as activ. f.

MODE = "poly" # "poly": uses log+exp activations, "standard": uses sigmoid activation, or "linear" for standard linear regression /w bias
CUSTOMNOTE = "DEBUGGING"
# output_dim = 1

# Learning algo
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.01

# *********************
dm = MyDataModule(
    batch_size=BATCH_SIZE,
    n=NUM_OBS,
    d=DIM,
    low=LOW,
    high=HIGH,
    funct=TO_LEARN,
    scale=SCALE,
)

model = PolyModel(
    input_dim=DIM,
    hidden_dim=HIDDEN_DIM,
    learning_rate=LEARNING_RATE,
    mode=MODE,
)

logger = pl.loggers.TensorBoardLogger(
    'logs', 
    name=f'{MODE}.{CUSTOMNOTE}.{TO_LEARN}.low-{int(LOW)}.high-{int(HIGH)}.'\
    f'nobs-{NUM_OBS}.dim-{DIM}.lrate-{LEARNING_RATE}.scale-{SCALE}',
    default_hp_metric=False,

)
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    # default_root_dir='ckpts',
    gpus=1,
    logger=logger,
    log_every_n_steps=100,
    # track_grad_norm=2,
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
