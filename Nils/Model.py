import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy().copy()


class LogActivation(nn.Module):
    def forward(self, x):
        return torch.log(x)
        return torch.log(torch.maximum(x, torch.tensor(1e-10)))


class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)


# Define model
class PolynomialNN(nn.Module):
    def __init__(self, nrvariables, n_monomials=1, exponent_bias=True):
        super(PolynomialNN, self).__init__()
        self.stack = nn.Sequential(
            LogActivation(),
            nn.Linear(nrvariables, n_monomials, bias=exponent_bias),
            ExpActivation(),
            nn.Linear(n_monomials, 1)
        )
        self.exponent_bias = exponent_bias

    def forward(self, x):
        x = self.stack(x)
        return x

    @property
    def exponent_layer(self):
        return self.stack[1]

    @property
    def coefficient_layer(self):
        return self.stack[3]

    def get_exponents(self):
        return tensor_to_numpy(self.stack[1].weight)

    def get_coefficients(self):
        coefficients = tensor_to_numpy(self.coefficient_layer.weight)
        if self.exponent_bias:
            coefficients *= np.exp(tensor_to_numpy(self.exponent_layer.bias))
        return coefficients

    def get_bias(self):
        return tensor_to_numpy(self.coefficient_layer.bias)

    def exponents_abs_(self):
        self.exponent_layer.weight.abs_()

    def force_non_negative_exponents_(self):
        with torch.no_grad():
            self.exponent_layer.weight.clamp_(0.)
