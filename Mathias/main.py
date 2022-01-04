# %%
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

# files
from models.poly import PolyModel
from models.standard import StandardModel
from datamodule import MyDataModule
from functions import polynomialf, sinf, cosinef, expf, logf, uniform_args, normal_args, linspace_args, truncated_normal

# Ignore warning in training when num_workers=0 (possible bottleneck)
import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# ************ LEARNING SIMPLE FUNCTIONS WITH 10 MONOMIALS/ NEURONS ***********************************************

# Instructions to reproduce the gridplot of the toy examples in our paper "Polynomial Feature Learning":

# 1. To reproduce the toy plots of the paper set seed to 42:
seed_everything(42, workers=True)

# 2. Choose function out of [polynomialf, sinf, cosinef, expf, logf] # Note: the polynomial function looks like this: 0.2*x**3 - 1.5*x**2 + 3.6*x - 2.5.
TARGET_FN = polynomialf

# 3. Specify model class here:
NN_ARCHITECTURE = PolyModel
# [PolyModel] for our Polynomial NN
# [StandardModel] for ReLU NN

# 4. Wait for training to end and use saved checkpoint "last.ckpt" later in Step 6.

# 5. Repeat with all other 9 combinations of functions and NN architectures.

# 6. When checkpoints for all functions and both network architectures are saved, specify their paths in the gridplotter.py file.

# 7. Run the gridplotter.py file. Done!

# ******************************************************************************************************************
# Hyperparameters
# Note: DONT CHANGE below parameters, if the goal is to reproduce the gridplot with the toy examples in our paper.

# Data
NUM_OBS = 1000 # 1000 used for gridplot in paper
DIM = 1 # 1 used for gridplot in paper
LOW = 1. # 1. used for gridplot in paper # if normal_args or truncated_normal corresponds to mean
HIGH = 5. # 5. used for gridplot in paper # if normal_args/ truncated_normal corresponds to std
SAMPLE_FN = uniform_args #uniform args used for gridplot in paper [uniform_args, normal_args, linspace_args, truncated-normal]
NOISE_LEVEL = 0.1 # 0.1 used for gridplot in paper

# Learning algo
HIDDEN_DIM = 10 # 10 used for gridplot in paper. number of neurons in hidden layer [if PolyModel = number of monomials, if StandardModel (ReLU) = size of hidden layer]
BATCH_SIZE = 128 # 128 used for gridplot in paper
NUM_EPOCHS = 300000 # 1
LEARNING_RATE = 1e-4


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

model = NN_ARCHITECTURE(
    input_dim=DIM,
    hidden_dim=HIDDEN_DIM,
    learning_rate=LEARNING_RATE,
)

logger = pl.loggers.TensorBoardLogger(
    'logs'
)

checkpoint_callback = ModelCheckpoint(
    every_n_epochs=1000000,
    save_last=True, #saves model at the end of training -> to load in gridplotter
    )

trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    gpus=1,
    logger=logger, #=logger or False
    check_val_every_n_epoch=10000,
    callbacks=checkpoint_callback, #default is after each training epoch
    # enable_checkpointing = False,
    num_sanity_val_steps=0,
)

# Fit the model
trainer.fit(model, dm)

