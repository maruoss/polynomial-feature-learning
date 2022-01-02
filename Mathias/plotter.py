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
        self.X_train = self.dm.X_train.detach().clone()
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

        # Unnormalize predictions
        y_pred = y_pred * self.dm.target_std + self.dm.target_mean
        y_pred_train = y_pred_train * self.dm.target_std + self.dm.target_mean
        
        
        # Create groundtruth over possibly larger domain
        y = self.target_fn(self.x)
        
        fig, ax = plt.subplots(1, 2, figsize = (14, 5))
        if self.model.__class__.__name__ == "PolyModel":
            ax[0].set_title(f"{self.target_fn.__name__[:-1].title()} Function learned with {len(self.model.l1.weight)} Monomials")
        elif self.model.__class__.__name__ == "StandardModel":
            ax[0].set_title(f"{self.target_fn.__name__[:-1].title()} Function learned with {len(self.model.l1.weight)} Neurons")
        ax[0].plot(self.x, y, label="groundtruth", color="red")
        ax[0].plot(self.x, y_pred.cpu(), label="learned function", color="orange")
        ax[0].scatter(self.X_train, self.dm.y_train_noisy, alpha=0.2, label="training set")

        ax[0].set_ylim(-5, 15)
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].legend()

        
        # Create groundtruth over possibly larger domain
        self.x = torch.linspace(self.X_train.min(), self.X_train.max(), 200).view(-1, 1)
        y = self.target_fn(self.x)

        if self.model.__class__.__name__ == "PolyModel":
            ax[1].set_title(f"{self.target_fn.__name__[:-1].title()} Function learned with {len(self.model.l1.weight)} Monomials")
        elif self.model.__class__.__name__ == "StandardModel":
            ax[1].set_title(f"{self.target_fn.__name__[:-1].title()} Function learned with {len(self.model.l1.weight)} Neurons")

        ax[1].plot(self.x, y, label="groundtruth", color="red")
        ax[1].plot(self.x, y_pred.cpu(), label="learned function", color="orange")
        ax[1].scatter(self.X_train, self.dm.y_train_noisy, alpha=0.2, label="training set")

        # ax[1].ylim(-5, 15)
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        ax[1].legend()
        # plt.autoscale(enable=False) # to autoscale axes if difference gets small
        # plt.show() # plt.show() before plt.savefig saves empty figure