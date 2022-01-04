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
from numpy.polynomial import Polynomial as T

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
NUM_EPOCHS = 300000
LEARNING_RATE = 1e-5

# Plotting options, oos = to plot testing out-of-sample points
LOW_OOS = 0.01 #only affects plots, needs oos=True
HIGH_OOS = 9. #only affects plots, needs oos=True

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
    every_n_epochs=1000000,
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
# # # Run learning rate finder
# lr_finder = trainer.tuner.lr_find(model, dm)

# # Results can be found in
# lr_finder.results

# # Plot with
# fig = lr_finder.plot(suggest=True)
# fig.show()
# %%
# Load Model
# model = PolyModel.load_from_checkpoint(r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias\logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-1e-05.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_3\checkpoints\last.ckpt")
# # model.load_state_dict(torch.load(r"C:\Users\Mathiass\Desktop\Experiments\polynomial-feature-learning\EXP1.POLYNOMIAL.10MONOMIALS.PolyModel.polynomialf.uniform_args.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.lrate-1e-05.batchsize-128\version_0\checkpoints\epoch=249999-step=1999999.ckpt"))
# print(model.hparams.learning_rate)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREDICTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainer.test(
    model, dm
)

# %%
# Show model predictions and save plot (creates 3 plots next to each other with different zoom levels)
plotter = predictions(dm, model, low_oos=LOW_OOS, high_oos=HIGH_OOS, target_fn=TARGET_FN)
plotter.plot()
plt.tight_layout()
path = os.path.join(logger.log_dir, f"final_predictions_{NUM_EPOCHS}.png")
plt.savefig(path, facecolor="white")
plt.show()

# print(f"X_test: {plotter.X_test}")
# print(f"y_pred_test: {plotter.y_pred_test}")

#
# Plot 3 Power Series plots each zoomed differently ***************************************************

# get X_train domain
X_train = dm.X_train
y_train = dm.y_train
# Fit polynomial
p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
_, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])

# Unnormalize
y_pred = y_pred * dm.target_std.item() + dm.target_mean.item()

# get extended domain
x = torch.linspace(LOW_OOS, HIGH_OOS, 200)
y = TARGET_FN(x)

fig, ax = plt.subplots(1, 3, figsize = (21, 5))

if TARGET_FN.__name__ == "polynomialf":
    function_name = "Polynomial Function"
elif TARGET_FN.__name__ == "sinf":
    function_name = "sin(x)"
elif TARGET_FN.__name__ == "cosinef":
    function_name = "cos(x)"
elif TARGET_FN.__name__ == "expf":
    function_name = "exp(x)"
elif TARGET_FN.__name__ == "logf":
    function_name = "log(x)"


ax[0].set_title(f"{function_name} fitted with Polynomials up to degree {HIDDEN_DIM}")

ax[0].plot(x, y, label="groundtruth", color="red")
ax[0].plot(x, y_pred, label="learned function", color="orange")
ax[0].scatter(X_train, dm.y_train_noisy, alpha=0.2, label="training set")

# ax[0].set_ylim([2*y.min().item(), 2*y.max().item()])
if  TARGET_FN.__name__ in ["sinf", "cosinef", "logf"]:
    ax[0].set_ylim(-4, 4) #fixed scale of plot
else:
    ax[0].set_ylim(-5, 55)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].legend()

# Subplot 2   
_, y_pred = p.linspace(200, domain=[0.01, 7.])

# Unnormalize
y_pred = y_pred * dm.target_std.item() + dm.target_mean.item()

# Create groundtruth over training domain
x = torch.linspace(0.01, 7., 200)
# Create groundtruth
y = TARGET_FN(x)

ax[1].set_title(f"{function_name} fitted with Polynomials up to degree {HIDDEN_DIM}")

ax[1].plot(x, y, label="groundtruth", color="red")
ax[1].plot(x, y_pred, label="learned function", color="orange")
ax[1].scatter(X_train, dm.y_train_noisy, alpha=0.2, label="training set")

# ax[0].set_ylim([2*y.min().item(), 2*y.max().item()])
if  TARGET_FN.__name__ in ["sinf", "cosinef", "logf"]:
    ax[1].set_ylim(-4, 4) #fixed scale of plot
else:
    ax[1].set_ylim(-5, 15)
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].legend()

# Subplot 3
_, y_pred = p.linspace(200, domain=[X_train.min(), X_train.max()])

# Unnormalize
y_pred = y_pred * dm.target_std.item() + dm.target_mean.item()

# Create groundtruth over training domain
x = torch.linspace(X_train.min(), X_train.max(), 200)
# Groundtruth
y = TARGET_FN(x)

ax[2].set_title(f"{function_name} fitted with Polynomials up to degree {HIDDEN_DIM}")

ax[2].plot(x, y, label="groundtruth", color="red")
ax[2].plot(x, y_pred, label="learned function", color="orange")
ax[2].scatter(X_train, dm.y_train_noisy, alpha=0.2, label="training set")

# ax[1].set_ylim([y.min().item(), y.max().item()])
if  TARGET_FN.__name__ in ["sinf", "cosinef"]:
    ax[2].set_ylim(-1.5, 1.5) #fixed scale of plot
elif TARGET_FN.__name__ in ["expf"]:
    ax[2].set_ylim([y.min().item() - 0.2, y.max().item()])
elif TARGET_FN.__name__ in ["logf"]:
    ax[2].set_ylim([y.min().item(), y.max().item() + 0.1])
else:
    ax[2].set_ylim([y.min().item(), y.max().item()])

ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
ax[2].legend()

plt.tight_layout()
path = os.path.join(logger.log_dir, "polynomial_fit.png")
plt.savefig(path, facecolor="white")
plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# GRIDPLOTTING

print("Read in paths...")
# Specify your absolute Model Paths here
ax1_polynomialf_polymodel = r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias\logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-0.0001.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"
ax2_polynomialf_standardmodel = r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias\logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-0.0001.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"

ax4_sinf_polymodel = r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias\logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-0.0001.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"
ax5_sinf_standardmodel = r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias\logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-0.0001.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"

ax7_cosinef_polymodel = r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias\logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-0.0001.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"
ax8_cosinef_standardmodel = r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias\logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-0.0001.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"

ax10_expf_polymodel = r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias\logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-0.0001.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"
ax11_expf_standardmodel = r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias\logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-0.0001.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"

ax13_logf_polymodel = r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias\logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-0.0001.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"
ax14_logf_standardmodel = r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias\logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-0.0001.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"

print("Done...")

# %%
print("Creating grid...")
# Domain that models were trained on
LOW = 1.
HIGH = 5.
# Set range x-axis to plot
LOW_OOS = 0.
HIGH_OOS = 7.
# Monomials = # Neurons = Degree Power Series
HIDDEN_DIM = 10

# Create x for groundtruth and predicted function
X_train = torch.linspace(LOW, HIGH, 200)
x = torch.linspace(LOW_OOS, HIGH_OOS, 200)

fig = plt.figure(figsize=(15, 15))
gs = fig.add_gridspec(5, 3, hspace=0, wspace=0)
(ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15) = gs.subplots(sharex='col', sharey='row')

# codomain to plot
ax1.set_ylim(-5, 15) # polynomial f.
ax4.set_ylim(-4, 4) # sine f.
ax7.set_ylim(-4, 4) # cosine f.
ax10.set_ylim(-5, 30) # exp f.
ax13.set_ylim(-4, 4) # log f.

# Fix x axis
ax1.set_xlim(0, 7.)
ax2.set_xlim(0, 7.)
ax3.set_xlim(0, 7.)

# titles
# fig.suptitle(f"Functions fitted with {HIDDEN_DIM} Monomials/ hidden Neurons/ Degree")
ax1.set_title(f"Polynomial NN with {HIDDEN_DIM} Monomials")
ax2.set_title(f"ReLU NN with {HIDDEN_DIM} Neurons")
ax3.set_title(f"Power Series up to degree {HIDDEN_DIM}")

ax1.set(ylabel="polynomial")
ax4.set(ylabel="sin(x)")
ax7.set(ylabel="cos(x)")
ax10.set(ylabel="exp(x-4)")
ax13.set(ylabel="log(x)")

# ax[0].tick_params(axis='y', rotation=90)

# ******************************************************************************
# Polynomial function
# Groundtruth Polynomial
y = 0.2*x**3 - 1.5*x**2 + 3.6*x - 2.5

# Load necessary files
target_mean = torch.load(f"y_train_noisy/MEAN.polynomialf.low{LOW}.high{HIGH}.pt")
target_std = torch.load(f"y_train_noisy/STD.polynomialf.low{LOW}.high{HIGH}.pt")
X_train = torch.load(f"X_train/uniform_args.low{LOW}.high{HIGH}.pt")
y_train_noisy = torch.load(f"y_train_noisy/polynomialf.low{LOW}.high{HIGH}.pt")


# ***** PolyNet *****
# Load trained model
# path1 = path.joinpath(relpath_polynomialf_polymodel)
model = model.load_from_checkpoint(ax1_polynomialf_polymodel)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean

# Plot everything
ax1.plot(x, y, label="groundtruth", color="red")
ax1.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax1.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# **** ReLUNet ****
# Load trained model
model = model.load_from_checkpoint(ax2_polynomialf_standardmodel)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean

ax2.plot(x, y, label="groundtruth", color="red")
ax2.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax2.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# **** Power Series Fitting ****
# Normalize y_train_noisy before fit as was done for other models
y_train = (y_train_noisy - target_mean) / target_std
# Fit
p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
_, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])

# Unnormalize Prediction
y_pred = y_pred * target_std.item() + target_mean.item()

ax3.plot(x, y, label="groundtruth", color="red")
ax3.plot(x, y_pred, label="learned function", color="orange")
ax3.scatter(X_train, y_train_noisy, alpha=0.2, label="training set", color="tab:blue")


# ************************************************************************************
# Sine function
# Groundtruth sine
y = torch.sin(x)

# Load necessary files
target_mean = torch.load(f"y_train_noisy/MEAN.sinf.low{LOW}.high{HIGH}.pt")
target_std = torch.load(f"y_train_noisy/STD.sinf.low{LOW}.high{HIGH}.pt")
X_train = torch.load(f"X_train/uniform_args.low{LOW}.high{HIGH}.pt")
y_train_noisy = torch.load(f"y_train_noisy/sinf.low{LOW}.high{HIGH}.pt")


# ***** PolyNet *****
# Load trained model
model = model.load_from_checkpoint(ax4_sinf_polymodel)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean

# Plot everything
ax4.plot(x, y, label="groundtruth", color="red")
ax4.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax4.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# **** ReLUNet ****
# Load trained model
model = model.load_from_checkpoint(ax5_sinf_standardmodel)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean

ax5.plot(x, y, label="groundtruth", color="red")
ax5.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax5.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# **** Power Series Fitting ****
# Normalize y_train_noisy before fit as was done for other models
y_train = (y_train_noisy - target_mean) / target_std
# Fit
p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
_, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])

# Unnormalize prediction
y_pred = y_pred * target_std.item() + target_mean.item()

ax6.plot(x, y, label="groundtruth", color="red")
ax6.plot(x, y_pred, label="learned function", color="orange")
ax6.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# ************************************************************************************
# cosine function
# Groundtruth cosine
y = torch.cos(x)

# Load necessary files
target_mean = torch.load(f"y_train_noisy/MEAN.cosinef.low{LOW}.high{HIGH}.pt")
target_std = torch.load(f"y_train_noisy/STD.cosinef.low{LOW}.high{HIGH}.pt")
X_train = torch.load(f"X_train/uniform_args.low{LOW}.high{HIGH}.pt")
y_train_noisy = torch.load(f"y_train_noisy/cosinef.low{LOW}.high{HIGH}.pt")


# ***** PolyNet *****
# Load trained model
model = model.load_from_checkpoint(ax7_cosinef_polymodel)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean

# Plot everything
ax7.plot(x, y, label="groundtruth", color="red")
ax7.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax7.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# **** ReLUNet ****
# Load trained model
model = model.load_from_checkpoint(ax8_cosinef_standardmodel)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean

ax8.plot(x, y, label="groundtruth", color="red")
ax8.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax8.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# **** Power Series Fitting ****
# Normalize y_train_noisy before fit as was done for other models
y_train = (y_train_noisy - target_mean) / target_std
# Fit
p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
_, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])

# Unnormalize prediction
y_pred = y_pred * target_std.item() + target_mean.item()

ax9.plot(x, y, label="groundtruth", color="red")
ax9.plot(x, y_pred, label="learned function", color="orange")
ax9.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# ***********************************************************************************
# exp function
# Groundtruth exp
y = torch.exp(x - 4)

# Load necessary files
target_mean = torch.load(f"y_train_noisy/MEAN.expf.low{LOW}.high{HIGH}.pt")
target_std = torch.load(f"y_train_noisy/STD.expf.low{LOW}.high{HIGH}.pt")
X_train = torch.load(f"X_train/uniform_args.low{LOW}.high{HIGH}.pt")
y_train_noisy = torch.load(f"y_train_noisy/expf.low{LOW}.high{HIGH}.pt")

# ***** PolyNet *****
# Load trained model
model = model.load_from_checkpoint(ax10_expf_polymodel)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean

# Plot everything
ax10.plot(x, y, label="groundtruth", color="red")
ax10.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax10.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# **** ReLUNet ****
# Load trained model
model = model.load_from_checkpoint(ax11_expf_standardmodel)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean

ax11.plot(x, y, label="groundtruth", color="red")
ax11.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax11.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# **** Power Series Fitting ****
# Normalize y_train_noisy before fit as was done for other models
y_train = (y_train_noisy - target_mean) / target_std
# Fit
p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
_, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])

# Unnormalize prediction
y_pred = y_pred * target_std.item() + target_mean.item()

ax12.plot(x, y, label="groundtruth", color="red")
ax12.plot(x, y_pred, label="learned function", color="orange")
ax12.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# ***********************************************************************************
# log function
# Groundtruth log
y = torch.log(x)

# Load necessary files
target_mean = torch.load(f"y_train_noisy/MEAN.logf.low{LOW}.high{HIGH}.pt")
target_std = torch.load(f"y_train_noisy/STD.logf.low{LOW}.high{HIGH}.pt")
X_train = torch.load(f"X_train/uniform_args.low{LOW}.high{HIGH}.pt")
y_train_noisy = torch.load(f"y_train_noisy/logf.low{LOW}.high{HIGH}.pt")

# ***** PolyNet *****
# Load trained model
model = model.load_from_checkpoint(ax13_logf_polymodel)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean

# Plot everything
ax13.plot(x, y, label="groundtruth", color="red")
ax13.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax13.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# **** ReLUNet ****
# Load trained model
model = model.load_from_checkpoint(ax14_logf_standardmodel)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean

ax14.plot(x, y, label="groundtruth", color="red")
ax14.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax14.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")


# **** Power Series Fitting ****
# Normalize y_train_noisy before fit as was done for other models
y_train = (y_train_noisy - target_mean) / target_std
# Fit
p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
_, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])

# Unnormalize prediction
y_pred = y_pred * target_std.item() + target_mean.item()

ax15.plot(x, y, label="groundtruth", color="red")
ax15.plot(x, y_pred, label="learned function", color="orange")
ax15.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

#* *************************************************************************************
# Make everything tight
gs.tight_layout(fig)
print("Done...")

print("Saving figure...")
# Save gridplot
fig.savefig(f"gridplot_{HIDDEN_DIM}Monomials", facecolor="white")
plt.show()

print("Done...")
print("Success!")





