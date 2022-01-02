import numpy as np
import torch
import matplotlib.pyplot as plt
from torch._C import device


class predictions:
    def __init__(self, 
                datamodule, 
                model,
                low_oos: float,
                high_oos: float, 
                oos: bool,
                target_fn,
                ):

        self.dm = datamodule
        self.model = model
        self.X_train = self.dm.X_train
        self.target_fn = target_fn
        self.device = model.device # in trainer will be cuda, outside will be cpu

        if oos:
            self.x = torch.linspace(low_oos, high_oos, 200).view(-1, 1)
            self.y = target_fn(self.x)
        else:
            self.x  = torch.from_numpy(np.linspace(self.X_train.min(), self.X_train.max(), 200)).view(-1, 1)
            self.y = target_fn(self.x)
        
            
    def plot(self):
        # clear all plots
        plt.close()
        plt.cla()
        plt.clf()
        # sns.set_style("darkgrid")

        with torch.no_grad():
            y_pred = self.model(self.x.to(torch.device(self.device)))
            y_pred_train = self.model(self.X_train.to(torch.device(self.device)))
        
        # Create groundtruth over possibly larger domain
        y = self.target_fn(self.x)

        # Unnormalize predictions
        y_pred = y_pred * self.dm.target_std + self.dm.target_mean
        y_pred_train = y_pred_train * self.dm.target_std + self.dm.target_mean
        

        if self.model.__class__.__name__ == "PolyModel":
            plt.title(f"{self.target_fn.__name__[:-1].title()} Function learned with {len(self.model.l1.weight)} Monomials")
        elif self.model.__class__.__name__ == "StandardModel":
            plt.title(f"{self.target_fn.__name__[:-1].title()} Function learned with {len(self.model.l1.weight)} Neurons")
        plt.plot(self.x, y, label="groundtruth", color="red")
        plt.plot(self.x, y_pred.cpu(), label="learned function", color="orange")
        plt.scatter(self.X_train, self.dm.y_train_noisy, alpha=0.2, label="training set")
        # plt.scatter(self.X_train, y_pred_train.cpu(), alpha=0.2, label="train prediction", marker=".")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        # plt.autoscale(enable=False) # to autoscale axes if difference gets small
        # plt.show() # plt.show() before plt.savefig saves empty figure