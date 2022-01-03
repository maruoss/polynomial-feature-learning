# %%
import time
import numpy as np
from pytorch_lightning import profiler
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
import os

# files
from plotter import predictions
from models.poly import PolyModel
from models.linear import LinearModel
from models.standard import StandardModel
from datamodule import MyDataModule

# %%
# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(42, workers=True)

# Ignore warning in training when num_workers=0 (possible bottleneck)
import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


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
    if hasattr(y, "shape"):
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = y.sum(axis=1, keepdim=True) + bias # sum across dimension and add single bias
    return y

# arbitrary polynomial function
def polynomialf(x: torch.Tensor) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs y
    in shape [obs, 1].
    """
    y = 0.2*x**3 - 1.5*x**2 + 3.6*x - 2.5
    if hasattr(y, "shape"):
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = y.sum(axis=1, keepdim=True) # sum across dimension
    return y

def sinf(x: torch.Tensor) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs y
    in shape [obs, 1].
    """
    y = torch.sin(x)
    if hasattr(y, "shape"):
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = y.sum(axis=1, keepdim=True) # sum across dimension
    return y

def cosinef(x: torch.Tensor) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs y
    in shape [obs, 1].
    """
    y = torch.cos(x)
    if hasattr(y, "shape"):
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = y.sum(axis=1, keepdim=True) # sum across dimension
    return y

def expf(x: torch.Tensor) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs y
    in shape [obs, 1].
    """
    y = torch.exp(x - 4) #shift exponent function to the right
    if hasattr(y, "shape"):
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = y.sum(axis=1, keepdim=True) # sum across dimension
    return y

def logf(x: torch.Tensor) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs y
    in shape [obs, 1].
    """
    y = torch.log(x)
    if hasattr(y, "shape"):
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = y.sum(axis=1, keepdim=True) # sum across dimension
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
    # datamatrix = torch.linspace(low, high-1, int((high-low))).view(-1, 1) #
    datamatrix = torch.linspace(low, high, n).view(-1, 1) #
    return datamatrix

def truncated_normal(n, d, min_val=1., low=0., high=1.):
    mean = low
    std = high
    normal_sample = lambda: torch.normal(mean, std, (n,))
    x = normal_sample()
    while (x < min_val).any():
        x_new = normal_sample()
        x[x < min_val] = x_new[x < min_val]
    return x.view(-1, 1) # Mathias adjustment, only for 1D

# %%
# Hyperparameters
# Synthetic data X of shape [NUM_OBS, DIM]
NUM_OBS = 1000
DIM = 1
LOW = 1. #=MEAN for normal_args
HIGH = 5. #=SD for normal_args
SAMPLE_FN = uniform_args

# Function to learn ("constantf", "linearf")
TARGET_FN = polynomialf

# Layer architecture
HIDDEN_DIM = 10      # equal to number of monomials if log/exp are used as activ. f.

# MODE = "poly" # "poly": uses log+exp activations, "standard": uses sigmoid activation, or "linear" for standard linear regression /w bias
CUSTOMNOTE = "DEBUGGING"
# output_dim = 1

# Learning algo
BATCH_SIZE = 128
NUM_EPOCHS = 250000
LEARNING_RATE = 1e-5

# Plotting options, oos = to plot testing out-of-sample points
LOW_OOS = 0.01 #only affects plots, needs oos=True
HIGH_OOS = 7. #only affects plots, needs oos=True

# CHECKPOINTS + SAVING PLOTS
PLOT_EVERY_N_EPOCHS = 10000 # has to be a multiple of "check_val_every_n_epoch"
TO_SAVE_PLOTS = True # whether to save plots to disk

# *********************
dm = MyDataModule(
    batch_size=BATCH_SIZE,
    sample_fn=SAMPLE_FN,
    n=NUM_OBS,  #keyword for sample_fn
    d=DIM,      #keyword for sample_fn
    low=LOW,    #keyword for sample_fn
    high=HIGH,  #keyword for sample_fn
    target_fn=TARGET_FN,
)
# Choose: PolyModel, StandardModel or LinearModel
model = PolyModel(
    input_dim=DIM,
    hidden_dim=HIDDEN_DIM,
    learning_rate=LEARNING_RATE,
    datamodule=dm,
    low_oos=LOW_OOS, 
    high_oos=HIGH_OOS, 
    target_fn=TARGET_FN,
    plot_every_n_epochs=PLOT_EVERY_N_EPOCHS,
    to_save_plots=TO_SAVE_PLOTS,
)

logger = pl.loggers.TensorBoardLogger(
    'logs', 
    name=f'{CUSTOMNOTE}.{model.__class__.__name__}.{TARGET_FN.__name__}.{SAMPLE_FN.__name__}.lrate-{LEARNING_RATE}.low-{LOW}.high-{HIGH}.'\
    f'nobs-{NUM_OBS}.dim-{DIM}.#monomials-{HIDDEN_DIM}.batchsize-{BATCH_SIZE}',
    default_hp_metric=False,

)

checkpoint_callback = ModelCheckpoint(
    # monitor="loss/val_loss",
    # filename="epoch={epoch:02d}-val_loss={loss/val_loss:.6f}",
    every_n_epochs=PLOT_EVERY_N_EPOCHS,
    save_last=True,
    # auto_insert_metric_name=False #to prevent {loss/val_loss} from creating subfolders because of /
    )

trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    # default_root_dir='ckpts',
    gpus=1,
    logger=logger, #=logger or False
    check_val_every_n_epoch=PLOT_EVERY_N_EPOCHS,
    callbacks=checkpoint_callback, #default is after each training epoch
    # enable_checkpointing = False,
    num_sanity_val_steps=0,
)
# 
trainer.fit(model, dm)

# %%
# Run learning rate finder
# lr_finder = trainer.tuner.lr_find(model, dm)

# # Results can be found in
# lr_finder.results

# # Plot with
# fig = lr_finder.plot(suggest=True)
# fig.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREDICTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainer.test(
    model, dm
)

# %%
# Show predictions and save plot
plotter = predictions(dm, model, low_oos=LOW_OOS, high_oos=HIGH_OOS, target_fn=TARGET_FN)
plotter.plot()
plt.tight_layout()
path = os.path.join(logger.log_dir, f"final_predictions_{NUM_EPOCHS}.png")
plt.savefig(path, facecolor="white")
plt.show()

# print(f"X_test: {plotter.X_test}")
# print(f"y_pred_test: {plotter.y_pred_test}")

#%%%
# # Show plot LAYER 1 WEIGHT/ EXPONENTS and save
# if model.__class__.__name__ == "PolyModel":
#     exponent_path = np.stack(model.exponent_path).squeeze(-1) #shape(2, 3)
#     coefficient_path = np.stack(model.coefficient_path).squeeze(-1) # shape (2, 3)
# elif model.__class__.__name__ == "StandardModel":
#     exponent_path = np.stack(model.l1_weights).squeeze(-1) #shape(2, 3)
#     coefficient_path = np.stack(model.l2_weights).squeeze(-1) # shape (2, 3)
# bias_path = np.stack(model.bias_path).squeeze(-1) # shape (2, 1)

# fig, ax = plt.subplots(1, 2, figsize = (14, 5))
# # Exponents
# for i in range(exponent_path.shape[-1]):
#     ax[0].plot(exponent_path[:, i])
# if TARGET_FN.__name__ == "constantf":
#     ax[0].axhline(0, label='Target Rank', c="red", ls="--")
# if TARGET_FN.__name__ == "linearf":
#     ax[0].axhline(1, label='Target Rank', c="red", ls="--")
# if TARGET_FN.__name__  in ["polynomialf", "polynomialf_noise"]:
#     ax[0].axhline(3, label='Target Rank', c="red", ls="--") #rank 3 monomial
#     ax[0].axhline(2, label='Target Rank', c="red", ls="--") #rank 2 monomial
#     ax[0].axhline(1, label='Target Rank', c="red", ls="--")      
#     ax[0].axhline(0, label='Target Rank', c="red", ls="--")

# ax[0].set_title("Learned Exponent Paths")
# ax[0].set_xlabel("Epoch")
# ax[0].set_ylabel("Exponent Value")


# # Coefficients
# for i in range(coefficient_path.shape[-1]):
#     ax[1].plot(coefficient_path[:, i])
                    
# ax[1].set_title("Learned Coefficient Paths")
# ax[1].set_xlabel("Epoch")
# ax[1].set_ylabel("Coefficient Value")


# # Save figure
# path = os.path.join(logger.log_dir, f"final_exponents_coefficients_{NUM_EPOCHS}.png")
# plt.savefig(path, facecolor="white")
# plt.show()


# %%