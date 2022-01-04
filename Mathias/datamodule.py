import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl


class MyDataModule(pl.LightningDataModule):
    def __init__(self, 
                batch_size: int,  
                sample_fn,
                n: float,
                d: float,
                low: float,
                high: float, 
                target_fn,
                noise_level:float,
                ):
        """
        Args:
        n = number of observations to generate
        d = dimensions to generate
        low = lower bound of data sampling
        high = upper bound of data sampling
        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

        # Train set
        self.X_train = sample_fn(n=n, d=d, low=low, high=high) # sample data
        self.y_train = target_fn(self.X_train) # create target values

        # Val set
        self.X_val = sample_fn(n= int(0.1*n), d=d, low=low, high=high) # sample data
        self.y_val = target_fn(self.X_val) # create target values
        # Test set
        self.X_test = sample_fn(n= int(0.1* n), d=d, low=low, high=high) # sample data
        self.y_test = target_fn(self.X_test) # create target values

        # Add Gaussian noise with SD = 0.1 * SD(y_train)
        self.y_train_noisy = self.add_noise(self.y_train)
        self.y_val = self.add_noise(self.y_val)
        self.y_test = self.add_noise(self.y_test)

        # Normalize targets
        self.target_mean = torch.mean(self.y_train_noisy)
        self.target_std = torch.std(self.y_train_noisy)
        self.y_train = self.normalize_target(self.y_train_noisy, self.target_mean, self.target_std)
        self.y_val = self.normalize_target(self.y_val, self.target_mean, self.target_std)
        self.y_test = self.normalize_target(self.y_test, self.target_mean, self.target_std)

    def setup(self, stage):
        
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
    
    def normalize_target(self, y, mean, std):
        return (y - mean)/std

    def add_noise(self, y):
        sd = self.hparams.noise_level * torch.std(y)
        y += torch.normal(0, sd, (len(y), 1))
        return y