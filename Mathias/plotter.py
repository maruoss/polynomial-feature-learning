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
                scale: bool, 
                oos: bool,
                target_fn,
                show_orig_scale: bool
                ):

        self.dm = datamodule
        self.model = model
        self.X_train = self.dm.X_train
        self.target_fn = target_fn
        self.show_orig_scale = show_orig_scale
        self.scale = scale
        self.device = model.device # in trainer will be cuda, outside will be cpu
        # self.oos = oos

        if oos:
            self.X_test = torch.linspace(low_oos, high_oos, 200).view(-1, 1)
            self.y_test = target_fn(self.X_test)
            if self.scale:
                self.X_test = self.dm.scaler.transform(self.X_test)
                self.X_test = torch.from_numpy(self.X_test).to(torch.float32)

        else:
            self.X_test  = self.dm.X_test
            self.y_test = self.dm.y_test
            
    def plot(self):
        # clear all plots
        plt.close()
        plt.cla()
        plt.clf()
        # sns.set_style("darkgrid")

        with torch.no_grad():
            self.y_pred_test = self.model(self.X_test.to(torch.device(self.device)))
            self.y_pred_train = self.model(self.X_train.to(torch.device(self.device)))

        if (self.show_orig_scale and self.scale):
            # transform back to original scale before plotting
            self.X_train = self.dm.scaler.inverse_transform(self.X_train)
            self.X_train = torch.from_numpy(self.X_train).to(torch.float32)
            self.X_test = self.dm.scaler.inverse_transform(self.X_test)
            self.X_test = torch.from_numpy(self.X_test).to(torch.float32)

        x = torch.cat([self.X_train, self.X_test], dim=0)
        x = torch.from_numpy(np.linspace(x.min(), x.max(), 200)).view(-1, 1) # linearf/ polynf expect this shape for .sum()

        # Create groundtruth over possibly larger domain
        y = self.target_fn(x)
        

        plt.plot(x, y, label="groundtruth")
        plt.scatter(self.X_train, self.y_pred_train.cpu(), c="orange", alpha=1., label="train prediction", marker=".", zorder=10.) #zorder defines which points have priority
        plt.scatter(self.X_test, self.y_pred_test.cpu(), c="green", alpha=0.5, label="test prediction", marker=".", zorder=5.)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        # plt.autoscale(enable=False) # to autoscale axes if difference gets small
        # plt.show() # plt.show() before plt.savefig saves empty figure