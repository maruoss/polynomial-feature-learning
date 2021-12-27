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
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 1),
            log_act(),
            #torch.nn.ReLU(),
            torch.nn.Linear(1, 1),
            exp_act()
        )
        self.layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.constant(m.weight, 0.2)
            m.bias.data.fill_(0)


    def forward(self, x):
        x = self.layers(x)
        return x
