import numpy as np
import torch
import matplotlib.pyplot as plt


class predictions:
    def __init__(self, 
                datamodule, 
                model,
                low_oos: float,
                high_oos: float, 
                target_fn,
                ):

        self.dm = datamodule
        self.model = model
        self.X_train = self.dm.X_train.detach().clone()
        self.target_fn = target_fn
        self.device = model.device # in trainer will be cuda, outside will be cpu

        # out of sample
        self.x = torch.linspace(low_oos, high_oos, 200).view(-1, 1)
        self.y = target_fn(self.x)

    def plot(self):
        # clear all plots
        plt.close()
        plt.cla()
        plt.clf()
        # sns.set_style("darkgrid")

        with torch.no_grad():
            y_pred = self.model(self.x.to(torch.device(self.device)))
        # Unnormalize predictions
        y_pred = y_pred * self.dm.target_std + self.dm.target_mean

        # Create groundtruth
        y = self.target_fn(self.x)
        
        fig, ax = plt.subplots(1, 3, figsize = (21, 5))

        if self.target_fn.__name__ == "polynomialf":
            function_name = "Polynomial Function"
        elif self.target_fn.__name__ == "sinf":
            function_name = "sin(x)"
        elif self.target_fn.__name__ == "cosinef":
            function_name = "cos(x)"
        elif self.target_fn.__name__ == "expf":
            function_name = "exp(x)"
        elif self.target_fn.__name__ == "logf":
            function_name = "log(x)"

        # Subplot 1
        if self.model.__class__.__name__ == "PolyModel":
            ax[0].set_title(f"{function_name} with {len(self.model.l1.weight)} Monomials after {self.model.current_epoch+1} Epochs")
        elif self.model.__class__.__name__ == "StandardModel":
            ax[0].set_title(f"{function_name} with {len(self.model.l1.weight)} Neurons after {self.model.current_epoch+1} Epochs")
        
        ax[0].plot(self.x, y, label="groundtruth", color="red")
        ax[0].plot(self.x, y_pred.cpu(), label="learned function", color="orange")
        ax[0].scatter(self.X_train, self.dm.y_train_noisy, alpha=0.2, label="training set")

        # ax[0].set_ylim([2*y.min().item(), 2*y.max().item()])
        if  self.target_fn.__name__ in ["sinf", "cosinef", "logf"]:
            ax[0].set_ylim(-4, 4) #fixed scale of plot
        else:
            ax[0].set_ylim(-5, 55)
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].legend()

        # Subplot 2   
        # Create groundtruth over training domain
        self.x = torch.linspace(0.01, 7., 200).view(-1, 1)
        # Predict on new out of sample set
        with torch.no_grad():
            y_pred = self.model(self.x.to(torch.device(self.device)))

        # Unnormalize predictions
        y_pred = y_pred * self.dm.target_std + self.dm.target_mean

        # Create groundtruth
        y = self.target_fn(self.x)
        
        if self.model.__class__.__name__ == "PolyModel":
            ax[1].set_title(f"{function_name} with {len(self.model.l1.weight)} Monomials after {self.model.current_epoch+1} Epochs")
        elif self.model.__class__.__name__ == "StandardModel":
            ax[1].set_title(f"{function_name} with {len(self.model.l1.weight)} Neurons after {self.model.current_epoch+1} Epochs")
        
        ax[1].plot(self.x, y, label="groundtruth", color="red")
        ax[1].plot(self.x, y_pred.cpu(), label="learned function", color="orange")
        ax[1].scatter(self.X_train, self.dm.y_train_noisy, alpha=0.2, label="training set")

        # ax[0].set_ylim([2*y.min().item(), 2*y.max().item()])
        if  self.target_fn.__name__ in ["sinf", "cosinef", "logf"]:
            ax[1].set_ylim(-4, 4) #fixed scale of plot
        else:
            ax[1].set_ylim(-5, 15)
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        ax[1].legend()

        
        # Subplot 3
        # Create groundtruth over training domain
        self.x = torch.linspace(self.X_train.min(), self.X_train.max(), 200).view(-1, 1)
        # Predict on new out of sample set
        with torch.no_grad():
            y_pred = self.model(self.x.to(torch.device(self.device)))
        # Unnormalize predictions
        y_pred = y_pred * self.dm.target_std + self.dm.target_mean

        # Groundtruth
        y = self.target_fn(self.x)

        if self.model.__class__.__name__ == "PolyModel":
            ax[2].set_title(f"{function_name} with {len(self.model.l1.weight)} Monomials after {self.model.current_epoch+1} Epochs")
        elif self.model.__class__.__name__ == "StandardModel":
            ax[2].set_title(f"{function_name} with {len(self.model.l1.weight)} Neurons after {self.model.current_epoch+1} Epochs")

        ax[2].plot(self.x, y, label="groundtruth", color="red")
        ax[2].plot(self.x, y_pred.cpu(), label="learned function", color="orange")
        ax[2].scatter(self.X_train, self.dm.y_train_noisy, alpha=0.2, label="training set")

        # ax[1].set_ylim([y.min().item(), y.max().item()])
        if  self.target_fn.__name__ in ["sinf", "cosinef"]:
            ax[2].set_ylim(-1.5, 1.5) #fixed scale of plot
        elif self.target_fn.__name__ in ["expf"]:
            ax[2].set_ylim([y.min().item() - 0.2, y.max().item()])
        elif self.target_fn.__name__ in ["logf"]:
            ax[2].set_ylim([y.min().item(), y.max().item() + 0.1])
        else:
            ax[2].set_ylim([y.min().item(), y.max().item()])

        ax[2].set_xlabel("x")
        ax[2].set_ylabel("y")
        ax[2].legend()
        # plt.autoscale(enable=False) # to autoscale axes if difference gets small
        # plt.show() # plt.show() before plt.savefig saves empty figure