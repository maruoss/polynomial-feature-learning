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
from models.standard_deep import StandardDeepModel
from datamodule import MyDataModule
from functions import polynomialf, sinf, cosinef, expf, logf, uniform_args, normal_args, linspace_args, truncated_normal

# %%
# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(42, workers=True)

# Ignore warning in training when num_workers=0 (possible bottleneck)
import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# %%
# Hyperparameters
# Synthetic data X of shape [NUM_OBS, DIM]
NUM_OBS = 1000
DIM = 1
LOW = 1. #=MEAN for normal_args
HIGH = 5. #=SD for normal_args
SAMPLE_FN = uniform_args
NOISE_LEVEL = 0.3

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
    noise_level=NOISE_LEVEL,
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
    name=f'{CUSTOMNOTE}.{model.__class__.__name__}.{TARGET_FN.__name__}.{SAMPLE_FN.__name__}.lrate-{LEARNING_RATE}.noise{NOISE_LEVEL}.low-{LOW}.high-{HIGH}.'\
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

# # Look at noise
# plt.scatter(dm.X_train, dm.y_train)


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



# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREDICTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# trainer.test(
#     model, dm
# )

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

# # get X_train domain
# X_train = dm.X_train
# y_train = dm.y_train
# # Fit polynomial
# p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
# _, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])

# # Unnormalize
# y_pred = y_pred * dm.target_std.item() + dm.target_mean.item()

# # get extended domain
# x = torch.linspace(LOW_OOS, HIGH_OOS, 200)
# y = TARGET_FN(x)

# fig, ax = plt.subplots(1, 3, figsize = (21, 5))

# if TARGET_FN.__name__ == "polynomialf":
#     function_name = "Polynomial Function"
# elif TARGET_FN.__name__ == "sinf":
#     function_name = "sin(x)"
# elif TARGET_FN.__name__ == "cosinef":
#     function_name = "cos(x)"
# elif TARGET_FN.__name__ == "expf":
#     function_name = "exp(x)"
# elif TARGET_FN.__name__ == "logf":
#     function_name = "log(x)"


# ax[0].set_title(f"{function_name} fitted with Polynomials up to degree {HIDDEN_DIM}")

# ax[0].plot(x, y, label="groundtruth", color="red")
# ax[0].plot(x, y_pred, label="learned function", color="orange")
# ax[0].scatter(X_train, dm.y_train_noisy, alpha=0.2, label="training set")

# # ax[0].set_ylim([2*y.min().item(), 2*y.max().item()])
# if  TARGET_FN.__name__ in ["sinf", "cosinef", "logf"]:
#     ax[0].set_ylim(-4, 4) #fixed scale of plot
# else:
#     ax[0].set_ylim(-5, 55)
# ax[0].set_xlabel("x")
# ax[0].set_ylabel("y")
# ax[0].legend()

# # Subplot 2   
# _, y_pred = p.linspace(200, domain=[0.01, 7.])

# # Unnormalize
# y_pred = y_pred * dm.target_std.item() + dm.target_mean.item()

# # Create groundtruth over training domain
# x = torch.linspace(0.01, 7., 200)
# # Create groundtruth
# y = TARGET_FN(x)

# ax[1].set_title(f"{function_name} fitted with Polynomials up to degree {HIDDEN_DIM}")

# ax[1].plot(x, y, label="groundtruth", color="red")
# ax[1].plot(x, y_pred, label="learned function", color="orange")
# ax[1].scatter(X_train, dm.y_train_noisy, alpha=0.2, label="training set")

# # ax[0].set_ylim([2*y.min().item(), 2*y.max().item()])
# if  TARGET_FN.__name__ in ["sinf", "cosinef", "logf"]:
#     ax[1].set_ylim(-4, 4) #fixed scale of plot
# else:
#     ax[1].set_ylim(-5, 15)
# ax[1].set_xlabel("x")
# ax[1].set_ylabel("y")
# ax[1].legend()

# # Subplot 3
# _, y_pred = p.linspace(200, domain=[X_train.min(), X_train.max()])

# # Unnormalize
# y_pred = y_pred * dm.target_std.item() + dm.target_mean.item()

# # Create groundtruth over training domain
# x = torch.linspace(X_train.min(), X_train.max(), 200)
# # Groundtruth
# y = TARGET_FN(x)

# ax[2].set_title(f"{function_name} fitted with Polynomials up to degree {HIDDEN_DIM}")

# ax[2].plot(x, y, label="groundtruth", color="red")
# ax[2].plot(x, y_pred, label="learned function", color="orange")
# ax[2].scatter(X_train, dm.y_train_noisy, alpha=0.2, label="training set")

# # ax[1].set_ylim([y.min().item(), y.max().item()])
# if  TARGET_FN.__name__ in ["sinf", "cosinef"]:
#     ax[2].set_ylim(-1.5, 1.5) #fixed scale of plot
# elif TARGET_FN.__name__ in ["expf"]:
#     ax[2].set_ylim([y.min().item() - 0.2, y.max().item()])
# elif TARGET_FN.__name__ in ["logf"]:
#     ax[2].set_ylim([y.min().item(), y.max().item() + 0.1])
# else:
#     ax[2].set_ylim([y.min().item(), y.max().item()])

# ax[2].set_xlabel("x")
# ax[2].set_ylabel("y")
# ax[2].legend()

# plt.tight_layout()
# path = os.path.join(logger.log_dir, "polynomial_fit.png")
# plt.savefig(path, facecolor="white")
# plt.show()
