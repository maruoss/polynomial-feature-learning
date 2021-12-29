import torch


class exp_act(torch.nn.Module):
    def __init__(self):
        super(exp_act, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        x = torch.exp(x)
        return x


class log_act(torch.nn.Module):
    def __init__(self):
        super(log_act, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        x = torch.add(x, 1)
        x = torch.log(x)
        return x


class MyModel(torch.nn.Module):
    def __init__(self, nrmonomials, nrvariables):
        super(MyModel, self).__init__()
        self.lin1 = torch.nn.Linear(1, nrvariables, bias=False)
        torch.nn.init.constant_(self.lin1.weight, 1)
        self.lin2 = torch.nn.Linear(nrvariables, nrmonomials, bias=False)
        torch.nn.init.normal_(self.lin2.weight, 0.1, 0.1)
        self.lin3 = torch.nn.Linear(nrmonomials, 1, bias=True)
        torch.nn.init.normal_(self.lin2.weight, 0.5, 0.1)
        self.layers = torch.nn.Sequential(
            # self.lin1,
            log_act(),
            # torch.nn.ReLU(),
            self.lin2,
            exp_act(),
            self.lin3
        )

    def forward(self, x):
        x = self.layers(x)
        return x
