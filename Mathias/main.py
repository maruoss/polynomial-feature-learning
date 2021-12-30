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
# Synthetic data X of shape [NUM_OBS, DIM]
NUM_OBS = 1600 # not used for linspace args [multiply by 0.64 for training size, 1600 -> 1024 training size]
DIM = 1
LOW = 2.5 #=MEAN for normal_args
HIGH = 1.0 #=SD for normal_args
SAMPLE_FN = truncated_normal

# Function to learn ("constantf", "linearf")
TARGET_FN = polynomialf
SCALE = False

# Layer architecture
HIDDEN_DIM = 3      # equal to number of monomials if log/exp are used as activ. f.

# MODE = "poly" # "poly": uses log+exp activations, "standard": uses sigmoid activation, or "linear" for standard linear regression /w bias
CUSTOMNOTE = "DEBUGGING"
# output_dim = 1

# Learning algo
BATCH_SIZE = 128
NUM_EPOCHS = 2000000
LEARNING_RATE = 1e-4

# Plotting options, oos = to plot testing out-of-sample points
OOS = True
LOW_OOS = 0. #only affects plots, needs oos=True
HIGH_OOS = 7. #only affects plots, needs oos=True
SHOW_ORIG_SCALE = False # #only affects plots, SCALE has to be True to have an effect

# CHECKPOINTS + SAVING PLOTS
PLOT_EVERY_N_EPOCHS = 200000 # has to be a multiple of "check_val_every_n_epoch"
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
    scale=SCALE,
)
# Choose: PolyModel, StandardModel or LinearModel
model = PolyModel(
    input_dim=DIM,
    hidden_dim=HIDDEN_DIM,
    learning_rate=LEARNING_RATE,
    datamodule=dm,
    low_oos=LOW_OOS, 
    high_oos=HIGH_OOS, 
    scale=SCALE,
    oos=OOS, 
    target_fn=TARGET_FN,
    show_orig_scale=SHOW_ORIG_SCALE, 
    plot_every_n_epochs=PLOT_EVERY_N_EPOCHS,
    to_save_plots=TO_SAVE_PLOTS,
)

logger = pl.loggers.TensorBoardLogger(
    'logs', 
    name=f'{CUSTOMNOTE}.{TARGET_FN.__name__}.{SAMPLE_FN.__name__}.low-{LOW}.high-{HIGH}.'\
    f'nobs-{NUM_OBS}.dim-{DIM}.#monomials-{HIDDEN_DIM}.lrate-{LEARNING_RATE}.batchsize-{BATCH_SIZE}.scale-{SCALE}',
    default_hp_metric=False,

)

checkpoint_callback = ModelCheckpoint(
    monitor="loss/val_loss",
    filename="epoch={epoch:02d}-val_loss={loss/val_loss:.6f}",
    every_n_epochs=1000,
    save_last=True,
    auto_insert_metric_name=False #to prevent {loss/val_loss} from creating subfolders because of /
    )

trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    # default_root_dir='ckpts',
    gpus=1,
    logger=logger, #=logger or False
    # log_every_n_steps=1,
    # flush_logs_every_n_steps=50000,
    check_val_every_n_epoch=1000,
    # track_grad_norm=2,
    callbacks=checkpoint_callback, #default is after each training epoch
    num_sanity_val_steps=0,
    # profiler="simple",
)

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
from plotter import predictions
plotter = predictions(dm, model, low_oos=0., high_oos=5., scale=SCALE, oos=True, target_fn=TARGET_FN, 
                        show_orig_scale=True)
fig = plt.figure(figsize=(7, 5))
plotter.plot()
plt.tight_layout()
path = os.path.join(logger.log_dir, f"final_predictions_{NUM_EPOCHS}.png")
plt.savefig(path, facecolor="white")
plt.show()

# print(f"X_test: {plotter.X_test}")
# print(f"y_pred_test: {plotter.y_pred_test}")

#%%%
# Show plot LAYER 1 WEIGHT/ EXPONENTS and save
exponent_path = np.stack(model.exponent_path).squeeze(-1) #shape(2, 3)
coefficient_path = np.stack(model.coefficient_path).squeeze(-1) # shape (2, 3)
bias_path = np.stack(model.bias_path).squeeze(-1) # shape (2, 1)

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
    ax[0].plot(np.full(NUM_EPOCHS+1, i+1), label=label, c=colors[i], ls="--")
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
    ax[1].plot(np.full(NUM_EPOCHS+1, coefficient), label=label, c=colors[degree], ls="--")
    label = f"Learned {degree}-th degree coefficient"
    ax[1].plot(path, label=label, c=colors[degree])
        
ax[1].set_title("Learned Coefficient Paths")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Coefficient Value")
ax[1].legend()


# Save figure
path = os.path.join(logger.log_dir, f"final_exponents_coefficients_{NUM_EPOCHS}.png")
plt.savefig(path, facecolor="white")
plt.show()




# %%
# fig = plt.figure(figsize=(7, 5))
# exponents = np.stack(model.exponent_path).squeeze(axis=-1)
# for i in range(exponents.shape[-1]): # minus index (training step) column in last column
#     plt.plot(exponents[:, i])
#     if TARGET_FN == constantf:
#         plt.axhline(0, label='Target Rank', c="red", ls="--")
#     if TARGET_FN == linearf:
#         plt.axhline(1, label='Target Rank', c="red", ls="--")
#     if TARGET_FN == polynomialf:
#         plt.axhline(3, label='Target Rank', c="red", ls="--") #rank 3 monomial
#         plt.axhline(2, label='Target Rank', c="red", ls="--") #rank 2 monomial
#         plt.axhline(1, label='Target Rank', c="red", ls="--")      
#         # plt.axhline(0, label='Target Rank', c="red", ls="--")
# plt.title("Learned Exponent Paths")
# plt.xlabel("Epoch")
# plt.ylabel("Exponent Value")
# plt.tight_layout()
# os.makedirs('plots', exist_ok=True)
# plt.savefig("plots/exponents_plot.png", facecolor="white")

# print(exponents)
# %%

# plt.plot(model.layer1weights[:, 2].cpu(), model.layer1weights[:, 1].cpu())
# plt.plot(model.layer1weights[:, 2].cpu(), model.layer1weights[:, 0].cpu())



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

