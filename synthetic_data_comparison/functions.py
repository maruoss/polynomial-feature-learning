
import torch
from pytorch_lightning import seed_everything

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(42, workers=True)


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
        y = y.sum(axis=1, keepdim=True) # sum across dimension
    return y

def sinf(x: torch.Tensor) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs y
    in shape [obs, 1].
    """
    y = torch.sin(x)
    if hasattr(y, "shape"):
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = y.sum(axis=1, keepdim=True) # sum across dimension
    return y

def cosinef(x: torch.Tensor) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs y
    in shape [obs, 1].
    """
    y = torch.cos(x)
    if hasattr(y, "shape"):
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = y.sum(axis=1, keepdim=True) # sum across dimension
    return y

def expf(x: torch.Tensor) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs y
    in shape [obs, 1].
    """
    y = torch.exp(x - 3) #shift exponent function to the right
    if hasattr(y, "shape"):
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = y.sum(axis=1, keepdim=True) # sum across dimension
    return y

def logf(x: torch.Tensor) -> torch.Tensor:
    """
    Takes array x of shape [obs, dim] as input and outputs y
    in shape [obs, 1].
    """
    y = torch.log(x)
    if hasattr(y, "shape"):
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = y.sum(axis=1, keepdim=True) # sum across dimension
    return y

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