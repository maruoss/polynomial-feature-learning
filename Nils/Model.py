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
    def __init__(self, polyrank):
        super(MyModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, polyrank, bias=True),
            log_act(),
            #torch.nn.ReLU(),
            torch.nn.Linear(polyrank, 1, bias=True),
            exp_act()
        )
        self.layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 2, 1)
            m.bias.data.fill_(0.1)


    def forward(self, x):
        x = self.layers(x)
        return x
