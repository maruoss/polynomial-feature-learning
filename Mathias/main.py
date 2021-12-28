# %%
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from argparse import ArgumentParser
from torchmetrics.functional import r2_score
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import default_rng
import pdb

# files
from plotter import predictions
from models.polymodel import PolyModel
from datamodule import MyDataModule

# %%
# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(42, workers=True)

# Ignore warning in training when num_workers=0 (possible bottleneck)
import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# %%

# y = y.sum(dim=1, keepdim=True) + 1.
# train_test_split(X, y, random_state=42,
#                 train_size=0.7, shuffle=True)

# %%
# Unit test randomness in splits
# a = MyDataModule(batch_size=8, n=100, d=1, low=0., high=10., funct="constantf", scale=True)
# a.setup(stage=None)
# a.example()


# %%
# Model
# %%
# constant function for y values
def constantf(x: torch.Tensor, y: torch.FloatTensor=3.) -> torch.Tensor:
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

# arbitrary polynomial function
def polynomialf(x: torch.Tensor) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs y
    in shape [obs, 1].
    """
    y = x**3 + 2*x**2 + x + 1
    y = y.sum(axis=1, keepdim=True) # sum across dimension and add single bias
    return y

# %%
# create x values
def uniform_args(n, d, low, high) -> torch.Tensor():
    """
    Create n values of dimension d uniformly between [low, high]
    Input: n - number of observations
           d - dimension of observation
           low/ high - lower/ upper bounds of uniform distr.
    Output: [n, d] torch.FloatTensor
    """
    datamatrix = torch.FloatTensor(n, d).uniform_(low, high)
    return datamatrix

def normal_args(n, d, low, high) -> torch.Tensor():
    """
    Create n values of dimension d normally with mean, sigma
    Input: n - number of observations
           d - dimension of observation
           low = mean
           high = std
    Output: [n, d] torch.FloatTensor
    """
    datamatrix = torch.FloatTensor(n, d).normal_(low, high)
    return datamatrix


def linspace_args(n, d, low, high) -> torch.Tensor():
    """
    Create (high-low+1) values of dimension 1
    Input: low - lower bound of linspace
           high - upper bound of linspace
    Output: [(high-low+1), 1] torch.FloatTensor
    """
    datamatrix = torch.linspace(low, high-1, int((high-low))).view(-1, 1) #
    return datamatrix

# %% Unit test
# Create dataset and groundtruths
# X = normal_args(n=3, d=2, low=0, high=1)
# print(X)
# # y = constantf(X, y=3.)
# y = linearf(X, 2, 10)
# y
# %%
# %%
# %%
# scaler = MinMaxScaler(feature_range=(1e-6, 1))
# print(type(X))
# print(X)
# X = scaler.fit_transform(X)
# type(X)
# type(torch.from_numpy(X))

# %%
# Hyperparameters
# Synthetic data X of shape [NUM_OBS, DIM] with values unif. cont. between LOW and HIGH
NUM_OBS = 10000
DIM = 1
LOW = 5.
HIGH = 10.
SAMPLE_FN = uniform_args

# Function to learn ("constantf", "linearf")
TARGET_FN = constantf
SCALE = False

# Layer architecture
HIDDEN_DIM = 2      # equal to number of monomials if log/exp are used as activ. f.

MODE = "poly" # "poly": uses log+exp activations, "standard": uses sigmoid activation, or "linear" for standard linear regression /w bias
CUSTOMNOTE = "DEBUGGING"
# output_dim = 1

# Learning algo
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.01

# Plotting options
OOS = True
LOW_OOS = 0.
HIGH_OOS = 10.
SHOW_ORIG_SCALE = False

# *********************
dm = MyDataModule(
    batch_size=BATCH_SIZE,
    sample_fn=SAMPLE_FN,
    n=NUM_OBS,  #keyword for sample_fn
    d=DIM,      #keyword for sample_fn
    low=LOW,    #keyword for sample_fn
    high=HIGH,  #keyword for sample_fn
    target_fn=TARGET_FN,
    scale=SCALE,
)

model = PolyModel(
    input_dim=DIM,
    hidden_dim=HIDDEN_DIM,
    learning_rate=LEARNING_RATE,
    mode=MODE,
    datamodule=dm,
    low_oos=LOW_OOS,
    high_oos=HIGH_OOS,
    scale=SCALE,
    oos=OOS,
    target_fn=TARGET_FN,
    show_orig_scale=SHOW_ORIG_SCALE,
)

logger = pl.loggers.TensorBoardLogger(
    'logs', 
    name=f'{MODE}.{CUSTOMNOTE}.{TARGET_FN.__name__}.{SAMPLE_FN.__name__}.low-{int(LOW)}.high-{int(HIGH)}.'\
    f'nobs-{NUM_OBS}.dim-{DIM}.#monomials-{HIDDEN_DIM}.lrate-{LEARNING_RATE}.batchsize-{BATCH_SIZE}.scale-{SCALE}',
    default_hp_metric=False,

)

# checkpoint_callback = ModelCheckpoint(
#     monitor="loss/val_loss",
#     # filename="best"
#     )

trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    # default_root_dir='ckpts',
    gpus=1,
    logger=logger,
    log_every_n_steps=100,
    # track_grad_norm=2,
    # callbacks=checkpoint_callback,
)
trainer.fit(model, dm)




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREDICTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainer.test(
    model, dm
)

# %%
# Plotter settings
from plotter import predictions
plotter = predictions(dm, model, low_oos=4., high_oos=15., scale=SCALE, oos=True, target_fn=TARGET_FN, 
                        show_orig_scale=SCALE) 
plotter.plot()

# print(f"X_test: {plotter.X_test}")
print(f"y_pred_test: {plotter.y_pred_test}")


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




# # %% Useful commands to debug
# # print(list(self.parameters()))



# # %%
# np.random.randint(0, 1000)
# # %%
# #seed_everything doesnt set seed for this one
# from numpy.random import default_rng
# rng = default_rng()

# rng.integers(0, 100, endpoint=False)


# # %%
# X = torch.FloatTensor(1, 10).uniform_(0, 1000000)
# # x.dtype
# X
# a = X.size(dim=1)

# torch.ones(a).reshape(-1, 1).shape
# # %%
# X = uniform_args(10, 10, 0, 1)
# X

# constantf(X, y=3.)

#%%

# a = CustomDataset()
# %%

# for i in range(10):
#     print(a[i])

