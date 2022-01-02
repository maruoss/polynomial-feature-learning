
import torch
import torch.nn as nn
import torch.nn.functional as F


# poly architecture
class Poly_Net(nn.Module):

    def __init__(self, input_dim, mon_dim, out_dim, b1, b2):
        super(Poly_Net, self).__init__()
        # input should be of size input_size, output mon_size. trained weights=e_nm
        self.lin_hidden = nn.Linear(input_dim, mon_dim, bias=b1)
        self.lin_hidden.weight.data.uniform_(0.0, 1.0)
        # output lin layer, trained weights=a_m
        # self.dropout = nn.Dropout(0.025)
        self.lin_output = nn.Linear(mon_dim, out_dim, bias=b2)
        # alternative weights: -1, 1
        self.lin_output.weight.data.uniform_(0.0, 1.0)

    def forward(self, x):
        return self.lin_output(torch.exp(self.lin_hidden(torch.log(x))))


# relu fully connected nn architecture
class Relu_Net(nn.Module):

    def __init__(self, input_dim, mon_dim, out_dim, b1, b2):
        super(Relu_Net, self).__init__()
        # input should be of size input_size, output mon_size. trained weights=e_nm
        self.lin_hidden = nn.Linear(input_dim, mon_dim, bias=b1)
        # self.lin_hidden.weight.data.uniform_(0.0, 1.0)
        # output lin layer, trained weights=a_m
        self.lin_output = nn.Linear(mon_dim, out_dim, bias=b2)
        # alternative weights: -1, 1
        # self.lin_output.weight.data.uniform_(0.0, 1.0)

    def forward(self, x):
        return self.lin_output(F.relu(self.lin_hidden(x)))
