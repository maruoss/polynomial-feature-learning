import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler


class MyDataModule(pl.LightningDataModule):
    def __init__(self, 
                batch_size: int, 
                # scale:bool, 
                sample_fn,
                n: float,
                d: float,
                low: float,
                high: float, 
                target_fn,
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
        # self.scale = scale

        # Create full dataset
        self.X_train = sample_fn(n=n, d=d, low=low, high=high) # sample data
        self.y_train = target_fn(self.X_train) # create target values
        
        self.X_val = sample_fn(n= int(0.1*n), d=d, low=low, high=high) # sample data
        self.y_val = target_fn(self.X_val) # create target values
        
        self.X_test = sample_fn(n= int(0.1* n), d=d, low=low, high=high) # sample data
        self.y_test = target_fn(self.X_test) # create target values

        # Normalize targets
        self.target_mean = torch.mean(self.y_train)
        self.target_std = torch.std(self.y_train)
        self.y_train = self.normalize_target(self.y_train, self.target_mean, self.target_std)
        self.y_val = self.normalize_target(self.y_val, self.target_mean, self.target_std)
        self.y_test = self.normalize_target(self.y_test, self.target_mean, self.target_std)

    def setup(self, stage):
        
        # # Normalize targets
        # self.target_mean = torch.mean(self.y_train)
        # self.target_std = torch.std(self.y_train)
        # self.y_train = self.normalize_target(self.y_train, self.target_mean, self.target_std)
        # self.y_val = self.normalize_target(self.y_val, self.target_mean, self.target_std)
        # self.y_test = self.normalize_target(self.y_test, self.target_mean, self.target_std)

        # if self.scale:
        #     self.scaler = MinMaxScaler(feature_range=(1, np.exp(1))) # SET SCALE RANGE -> 0 IS NOT SUITABLE FOR LN transform -> -inf
        #     assert self.X_train.dtype == self.X_val.dtype == self.X_test.dtype
        #     datatype = self.X_train.dtype
        #     # Scale train, val, testset
        #     self.X_train = self.scaler.fit_transform(self.X_train)
        #     self.X_val = self.scaler.transform(self.X_val)
        #     self.X_test = self.scaler.transform(self.X_test)
        #     self.X_train = torch.from_numpy(self.X_train).to(datatype)
        #     self.X_val = torch.from_numpy(self.X_val).to(datatype)
        #     self.X_test = torch.from_numpy(self.X_test).to(datatype)

        
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