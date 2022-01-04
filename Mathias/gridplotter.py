# %%
import torch
from pytorch_lightning import seed_everything
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.polynomial import Polynomial as T
import os
from pathlib import Path

# files
from models.poly import PolyModel
from models.standard import StandardModel
from functions import polynomialf, sinf, cosinef, expf, logf, uniform_args

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(42, workers=True)

# %%
# Get or set project folder
# project_path = r"C:\Users\Mathiass\Documents\Projects\polynomial-feature-learning\Mathias"
project_path = os.getcwd()
# Convert into os indep. path
project_path = Path(project_path)


# # Unit Test
# # Dont use slashes at the start! On Windows: subpath with r"", on Linux subpath with forward slashes ".../file.ckpt"
# to_load = r"logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-1e-05.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"
# libdir = Path.joinpath(project_path, to_load)
# model = PolyModel.load_from_checkpoint(libdir)
# model

#%%
# Specify ckpt paths here:
ax1_polynomialf_polymodel = r"logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-1e-05.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"
ax2_polynomialf_standardmodel = r"logs\DEBUGGING.StandardModel.polynomialf.uniform_args.lrate-0.0001.noise0.1.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"

ax4_sinf_polymodel = r"logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-1e-05.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"
ax5_sinf_standardmodel = r"logs\DEBUGGING.StandardModel.polynomialf.uniform_args.lrate-0.0001.noise0.1.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"

ax7_cosinef_polymodel = r"logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-1e-05.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"
ax8_cosinef_standardmodel = r"logs\DEBUGGING.StandardModel.polynomialf.uniform_args.lrate-0.0001.noise0.1.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"

ax10_expf_polymodel = r"logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-1e-05.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"
ax11_expf_standardmodel = r"logs\DEBUGGING.StandardModel.polynomialf.uniform_args.lrate-0.0001.noise0.1.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"

ax13_logf_polymodel = r"logs\DEBUGGING.PolyModel.cosinef.uniform_args.lrate-1e-05.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"
ax14_logf_standardmodel = r"logs\DEBUGGING.StandardModel.polynomialf.uniform_args.lrate-0.0001.noise0.1.low-1.0.high-5.0.nobs-1000.dim-1.#monomials-10.batchsize-128\version_0\checkpoints\last.ckpt"

print("Read in paths successfully!")

# %%
print("Creating grid...")

# Specify domain that models were trained on
LOW = 1.
HIGH = 5.
# Specify x-axis to plot/ predict out of sample
LOW_OOS = 0.01
HIGH_OOS = 7.
# Specify used Monomials = # Neurons = Degree Power Series
# and used noise
HIDDEN_DIM = 10
NOISE_LEVEL = 0.1

# Create x for groundtruth and predicted function
x_insample = torch.linspace(LOW, HIGH, 200)
x = torch.linspace(LOW_OOS, HIGH_OOS, 200)

# Set fontsize
matplotlib.rcParams.update({'font.size': 16})
# Create grid
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
ax10.set(ylabel="exp(x-3)")
ax13.set(ylabel="log(x)")

# ax[0].tick_params(axis='y', rotation=90)

# ******************************************************************************
# Polynomial function
mse_losses_polynomial = []

# Groundtruth Polynomial
y = 0.2*x**3 - 1.5*x**2 + 3.6*x - 2.5
y_insample = 0.2*x_insample**3 - 1.5*x_insample**2 + 3.6*x_insample - 2.5

# Load necessary files
target_mean = torch.load(f"y_train_noisy/MEAN.polynomialf.low{LOW}.high{HIGH}.pt")
target_std = torch.load(f"y_train_noisy/STD.polynomialf.low{LOW}.high{HIGH}.pt")
X_train = torch.load(f"X_train/uniform_args.low{LOW}.high{HIGH}.pt") # the same for all functions
y_train_noisy = torch.load(f"y_train_noisy/polynomialf.low{LOW}.high{HIGH}.pt")


# ***** PolyNet *****
# Load trained model
# path1 = path.joinpath(relpath_polynomialf_polymodel)
to_load = Path.joinpath(project_path, ax1_polynomialf_polymodel)
model = PolyModel.load_from_checkpoint(to_load)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))
    y_pred_insample = model(x_insample.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean
y_pred_insample = y_pred_insample * target_std + target_mean

# Plot everything
ax1.plot(x, y, label="groundtruth", color="red")
ax1.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax1.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(y_pred.squeeze(), y)
ins_loss = nn.functional.mse_loss(y_pred_insample.squeeze(), y_insample)
# Append to list
mse_losses_polynomial.append((("oos_polymodel"), oos_loss.item()))
mse_losses_polynomial.append((("ins_polymodel"), ins_loss.item()))


# **** ReLUNet ****
# Load trained model
to_load = Path.joinpath(project_path, ax2_polynomialf_standardmodel)
model = StandardModel.load_from_checkpoint(to_load)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))
    y_pred_insample = model(x_insample.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean
y_pred_insample = y_pred_insample * target_std + target_mean

ax2.plot(x, y, label="groundtruth", color="red")
ax2.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax2.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(y_pred.squeeze(), y)
ins_loss = nn.functional.mse_loss(y_pred_insample.squeeze(), y_insample)
# Append to list
mse_losses_polynomial.append((("oos_standardmodel"), oos_loss.item()))
mse_losses_polynomial.append((("ins_standardmodel"), ins_loss.item()))

# **** Power Series Fitting ****
# Normalize y_train_noisy before fit as was done for other models
y_train = (y_train_noisy - target_mean) / target_std
# Fit
p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
_, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])
_, y_pred_insample = p.linspace(200, domain=[LOW, HIGH])

# Unnormalize Prediction
y_pred = y_pred * target_std.item() + target_mean.item()
y_pred_insample = y_pred_insample * target_std.item() + target_mean.item()

ax3.plot(x, y, label="groundtruth", color="red")
ax3.plot(x, y_pred, label="learned function", color="orange")
ax3.scatter(X_train, y_train_noisy, alpha=0.2, label="training set", color="tab:blue")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(torch.from_numpy(y_pred.squeeze()).to(torch.float32), y)
ins_loss = nn.functional.mse_loss(torch.from_numpy(y_pred_insample.squeeze()).to(torch.float32), y_insample)
# # Append to list
mse_losses_polynomial.append((("oos_powerseries"), oos_loss.item()))
mse_losses_polynomial.append((("ins_powerseries"), ins_loss.item()))

# %%
# ************************************************************************************
# Sine function
mse_losses_sine = []
# Groundtruth sine
y = torch.sin(x)
y_insample = torch.sin(x_insample)

# Load necessary files
target_mean = torch.load(f"y_train_noisy/MEAN.sinf.low{LOW}.high{HIGH}.pt")
target_std = torch.load(f"y_train_noisy/STD.sinf.low{LOW}.high{HIGH}.pt")
X_train = torch.load(f"X_train/uniform_args.low{LOW}.high{HIGH}.pt")
y_train_noisy = torch.load(f"y_train_noisy/sinf.low{LOW}.high{HIGH}.pt")


# ***** PolyNet *****
# Load trained model
to_load = Path.joinpath(project_path, ax4_sinf_polymodel)
model = PolyModel.load_from_checkpoint(to_load)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))
    y_pred_insample = model(x_insample.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean
y_pred_insample = y_pred_insample * target_std + target_mean

# Plot everything
ax4.plot(x, y, label="groundtruth", color="red")
ax4.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax4.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(y_pred.squeeze(), y)
ins_loss = nn.functional.mse_loss(y_pred_insample.squeeze(), y_insample)
# Append to list
mse_losses_sine.append((("oos_polymodel"), oos_loss.item()))
mse_losses_sine.append((("ins_polymodel"), ins_loss.item()))


# **** ReLUNet ****
# Load trained model
to_load = Path.joinpath(project_path, ax5_sinf_standardmodel)
model = StandardModel.load_from_checkpoint(to_load)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))
    y_pred_insample = model(x_insample.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean
y_pred_insample = y_pred_insample * target_std + target_mean

ax5.plot(x, y, label="groundtruth", color="red")
ax5.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax5.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(y_pred.squeeze(), y)
ins_loss = nn.functional.mse_loss(y_pred_insample.squeeze(), y_insample)
# Append to list
mse_losses_sine.append((("oos_standardmodel"), oos_loss.item()))
mse_losses_sine.append((("ins_standardmodel"), ins_loss.item()))


# **** Power Series Fitting ****
# Normalize y_train_noisy before fit as was done for other models
y_train = (y_train_noisy - target_mean) / target_std
# Fit
p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
_, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])
_, y_pred_insample = p.linspace(200, domain=[LOW, HIGH])

# Unnormalize prediction
y_pred = y_pred * target_std.item() + target_mean.item()
y_pred_insample = y_pred_insample * target_std.item() + target_mean.item()

ax6.plot(x, y, label="groundtruth", color="red")
ax6.plot(x, y_pred, label="learned function", color="orange")
ax6.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(torch.from_numpy(y_pred.squeeze()).to(torch.float32), y)
ins_loss = nn.functional.mse_loss(torch.from_numpy(y_pred_insample.squeeze()).to(torch.float32), y_insample)
# # Append to list
mse_losses_sine.append((("oos_powerseries"), oos_loss.item()))
mse_losses_sine.append((("ins_powerseries"), ins_loss.item()))


# ************************************************************************************
# cosine function
mse_losses_cosine = []
# Groundtruth cosine
y = torch.cos(x)
y_insample = torch.cos(x_insample)

# Load necessary files
target_mean = torch.load(f"y_train_noisy/MEAN.cosinef.low{LOW}.high{HIGH}.pt")
target_std = torch.load(f"y_train_noisy/STD.cosinef.low{LOW}.high{HIGH}.pt")
X_train = torch.load(f"X_train/uniform_args.low{LOW}.high{HIGH}.pt")
y_train_noisy = torch.load(f"y_train_noisy/cosinef.low{LOW}.high{HIGH}.pt")


# ***** PolyNet *****
# Load trained model
to_load = Path.joinpath(project_path, ax7_cosinef_polymodel)
model = PolyModel.load_from_checkpoint(to_load)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))
    y_pred_insample = model(x_insample.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean
y_pred_insample = y_pred_insample * target_std + target_mean

# Plot everything
ax7.plot(x, y, label="groundtruth", color="red")
ax7.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax7.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(y_pred.squeeze(), y)
ins_loss = nn.functional.mse_loss(y_pred_insample.squeeze(), y_insample)
# Append to list
mse_losses_cosine.append((("oos_polymodel"), oos_loss.item()))
mse_losses_cosine.append((("ins_polymodel"), ins_loss.item()))


# **** ReLUNet ****
# Load trained model
to_load = Path.joinpath(project_path, ax8_cosinef_standardmodel)
model = StandardModel.load_from_checkpoint(to_load)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))
    y_pred_insample = model(x_insample.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean
y_pred_insample = y_pred_insample * target_std + target_mean

ax8.plot(x, y, label="groundtruth", color="red")
ax8.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax8.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(y_pred.squeeze(), y)
ins_loss = nn.functional.mse_loss(y_pred_insample.squeeze(), y_insample)
# Append to list
mse_losses_cosine.append((("oos_standardmodel"), oos_loss.item()))
mse_losses_cosine.append((("ins_standardmodel"), ins_loss.item()))


# **** Power Series Fitting ****
# Normalize y_train_noisy before fit as was done for other models
y_train = (y_train_noisy - target_mean) / target_std
# Fit
p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
_, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])
_, y_pred_insample = p.linspace(200, domain=[LOW, HIGH])

# Unnormalize prediction
y_pred = y_pred * target_std.item() + target_mean.item()
y_pred_insample = y_pred_insample * target_std.item() + target_mean.item()

ax9.plot(x, y, label="groundtruth", color="red")
ax9.plot(x, y_pred, label="learned function", color="orange")
ax9.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(torch.from_numpy(y_pred.squeeze()).to(torch.float32), y)
ins_loss = nn.functional.mse_loss(torch.from_numpy(y_pred_insample.squeeze()).to(torch.float32), y_insample)
# # Append to list
mse_losses_cosine.append((("oos_powerseries"), oos_loss.item()))
mse_losses_cosine.append((("ins_powerseries"), ins_loss.item()))


# ***********************************************************************************
# exp function
mse_losses_exp = []
# Groundtruth exp
y = torch.exp(x - 3)
y_insample = torch.exp(x - 3)

# Load necessary files
target_mean = torch.load(f"y_train_noisy/MEAN.expf.low{LOW}.high{HIGH}.pt")
target_std = torch.load(f"y_train_noisy/STD.expf.low{LOW}.high{HIGH}.pt")
X_train = torch.load(f"X_train/uniform_args.low{LOW}.high{HIGH}.pt")
y_train_noisy = torch.load(f"y_train_noisy/expf.low{LOW}.high{HIGH}.pt")

# ***** PolyNet *****
# Load trained model
to_load = Path.joinpath(project_path, ax10_expf_polymodel)
model = PolyModel.load_from_checkpoint(to_load)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))
    y_pred_insample = model(x_insample.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean
y_pred_insample = y_pred_insample * target_std + target_mean

# Plot everything
ax10.plot(x, y, label="groundtruth", color="red")
ax10.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax10.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(y_pred.squeeze(), y)
ins_loss = nn.functional.mse_loss(y_pred_insample.squeeze(), y_insample)
# Append to list
mse_losses_exp.append((("oos_polymodel"), oos_loss.item()))
mse_losses_exp.append((("ins_polymodel"), ins_loss.item()))

# **** ReLUNet ****
# Load trained model
to_load = Path.joinpath(project_path, ax11_expf_standardmodel)
model = StandardModel.load_from_checkpoint(to_load)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))
    y_pred_insample = model(x_insample.to(torch.device(model.device)))  

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean
y_pred_insample = y_pred_insample * target_std + target_mean

ax11.plot(x, y, label="groundtruth", color="red")
ax11.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax11.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(y_pred.squeeze(), y)
ins_loss = nn.functional.mse_loss(y_pred_insample.squeeze(), y_insample)
# Append to list
mse_losses_exp.append((("oos_standardmodel"), oos_loss.item()))
mse_losses_exp.append((("ins_standardmodel"), ins_loss.item()))


# **** Power Series Fitting ****
# Normalize y_train_noisy before fit as was done for other models
y_train = (y_train_noisy - target_mean) / target_std
# Fit
p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
_, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])
_, y_pred_insample = p.linspace(200, domain=[LOW, HIGH])

# Unnormalize prediction
y_pred = y_pred * target_std.item() + target_mean.item()
y_pred_insample = y_pred_insample * target_std.item() + target_mean.item()

ax12.plot(x, y, label="groundtruth", color="red")
ax12.plot(x, y_pred, label="learned function", color="orange")
ax12.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(torch.from_numpy(y_pred.squeeze()).to(torch.float32), y)
ins_loss = nn.functional.mse_loss(torch.from_numpy(y_pred_insample.squeeze()).to(torch.float32), y_insample)
# # Append to list
mse_losses_exp.append((("oos_powerseries"), oos_loss.item()))
mse_losses_exp.append((("ins_powerseries"), ins_loss.item()))
# ***********************************************************************************
# log function
mse_losses_log = []
# Groundtruth log
y = torch.log(x)
y_insample = torch.log(x_insample)

# Load necessary files
target_mean = torch.load(f"y_train_noisy/MEAN.logf.low{LOW}.high{HIGH}.pt")
target_std = torch.load(f"y_train_noisy/STD.logf.low{LOW}.high{HIGH}.pt")
X_train = torch.load(f"X_train/uniform_args.low{LOW}.high{HIGH}.pt")
y_train_noisy = torch.load(f"y_train_noisy/logf.low{LOW}.high{HIGH}.pt")

# ***** PolyNet *****
# Load trained model
to_load = Path.joinpath(project_path, ax13_logf_polymodel)
model = PolyModel.load_from_checkpoint(to_load)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))
    y_pred_insample = model(x_insample.to(torch.device(model.device)))

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean
y_pred_insample = y_pred_insample * target_std + target_mean

# Plot everything
ax13.plot(x, y, label="groundtruth", color="red")
ax13.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax13.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(y_pred.squeeze(), y)
ins_loss = nn.functional.mse_loss(y_pred_insample.squeeze(), y_insample)
# Append to list
mse_losses_log.append((("oos_polymodel"), oos_loss.item()))
mse_losses_log.append((("ins_polymodel"), ins_loss.item()))


# **** ReLUNet ****
# Load trained model
to_load = Path.joinpath(project_path, ax14_logf_standardmodel)
model = StandardModel.load_from_checkpoint(to_load)
with torch.no_grad():
    y_pred = model(x.to(torch.device(model.device)))
    y_pred_insample = model(x_insample.to(torch.device(model.device)))  

# Unnormalize predictions
y_pred = y_pred * target_std + target_mean
y_pred_insample = y_pred_insample * target_std + target_mean

ax14.plot(x, y, label="groundtruth", color="red")
ax14.plot(x, y_pred.cpu(), label="learned function", color="orange")
ax14.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(y_pred.squeeze(), y)
ins_loss = nn.functional.mse_loss(y_pred_insample.squeeze(), y_insample)
# Append to list
mse_losses_log.append((("oos_standardmodel"), oos_loss.item()))
mse_losses_log.append((("ins_standardmodel"), ins_loss.item()))

# **** Power Series Fitting ****
# Normalize y_train_noisy before fit as was done for other models
y_train = (y_train_noisy - target_mean) / target_std
# Fit
p = T.fit(X_train.squeeze(), y_train.squeeze(), deg=HIDDEN_DIM)
_, y_pred = p.linspace(200, domain=[LOW_OOS, HIGH_OOS])
_, y_pred_insample = p.linspace(200, domain=[LOW, HIGH])

# Unnormalize prediction
y_pred = y_pred * target_std.item() + target_mean.item()
y_pred_insample = y_pred_insample * target_std.item() + target_mean.item()

ax15.plot(x, y, label="groundtruth", color="red")
ax15.plot(x, y_pred, label="learned function", color="orange")
ax15.scatter(X_train, y_train_noisy, alpha=0.2, label="training set")

# Calculate mse loss in sample (train) and out of sample (test)
oos_loss = nn.functional.mse_loss(torch.from_numpy(y_pred.squeeze()).to(torch.float32), y)
ins_loss = nn.functional.mse_loss(torch.from_numpy(y_pred_insample.squeeze()).to(torch.float32), y_insample)
# # Append to list
mse_losses_log.append((("oos_powerseries"), oos_loss.item()))
mse_losses_log.append((("ins_powerseries"), ins_loss.item()))

#* *************************************************************************************
# Make everything tight
gs.tight_layout(fig)
print("Done...")

# %%
print("Saving mse errors to csv")
import csv
total_list = [mse_losses_polynomial, mse_losses_sine, mse_losses_cosine, mse_losses_exp, mse_losses_log]
names = ["polynomial function:", "sine function:", "cosine function:", "exp function:", "log function:"]
with open(f'mse_losses_{HIDDEN_DIM}_noise{NOISE_LEVEL}.csv','w', newline='', encoding="utf-8") as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['domain_model','mse_loss'])
    for i, data in enumerate(total_list):
        csv_out.writerow([])
        csv_out.writerow([names[i]])
        csv_out.writerow([])
        for row in data:
            csv_out.writerow(row)

print("Done.")


print("Saving figure...")
# Save gridplot
fig.savefig(f"gridplot_{HIDDEN_DIM}Monomials", facecolor="white")
plt.show()

print("Done...")
print("Success!")
