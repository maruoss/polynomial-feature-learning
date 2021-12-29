import sklearn
from sklearn import preprocessing
from sklearn import model_selection
import torch
import scipy
import pandas as pd
from tqdm import tqdm
import numpy as np
import Model
from matplotlib import pyplot


def prepdata(x):
    # z_scores = x.apply(scipy.stats.zscore, nan_policy='omit')
    # x.where(abs(z_scores) < 3, inplace=True)
    # x = x.fillna(x.mean())
    # scaler = preprocessing.StandardScaler()
    # scaler.fit(x)
    # x = scaler.transform(x)
    return x


def compute_test_accuracy(model, val_loader, loss_fn, device):
    accuracy = []
    for batch in tqdm(val_loader, leave=False):
        X, label = batch
        outputs = model(X)
        acc = loss_fn(outputs, label)
        # print(acc)
        accuracy.append(acc)
    accuracy = torch.tensor(accuracy, device=device, dtype=torch.float).to(device)
    return torch.mean(accuracy)


def gen_func(x):
    y =  5*(x**4) + 2
    return y

def gen_data():
    mylen = 1000
    x_train =  np.reshape(np.arange(0, mylen)/mylen, (-1, 1))
    y_train =  gen_func(x_train)
    return x_train, y_train

nrmonomials = 4
nrvariables = 1
print("Starting")
x_train, y_train = gen_data()
print("Generated data")
x_train_orig = x_train
y_train_orig = y_train

x_train = prepdata(x_train)
print("preprocessing complete")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device is {device}.")


x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=0.1)
print("data splitting complete")

x_train_orig = torch.tensor(x_train_orig, device=device, dtype=torch.float).to(device)
x_train = torch.tensor(x_train, device=device, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, device=device, dtype=torch.float).to(device)
x_val = torch.tensor(x_val, device=device, dtype=torch.float).to(device)
y_val = torch.tensor(y_val, device=device, dtype=torch.float).to(device)

print("tensorification complete")
# print(x_train)

batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

model = Model.MyModel(nrmonomials, nrvariables).to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 50000

layer1weights = []
layer2weights = []
trainlosses = []

def train(model, train_loader, validation_loader, epochs, device):
    for epoch in range(epochs):
        iteration_loss = 0.
        for batch in tqdm(train_loader, leave=False):
            # print(batch)
            x, label = batch
            #print(batch)
            optimizer.zero_grad()
            out = model(x)
            # print(out.squeeze().shape, label.squeeze().shape)
            all_linear2_params = torch.cat([x.view(-1) for x in model.lin2.parameters()])
            all_linear3_params = torch.cat([x.view(-1) for x in model.lin3.parameters()])
            l1_regularization = (0.01 * torch.norm(all_linear2_params, 1))+(0.01 * torch.norm(all_linear3_params, 1))
            loss = loss_fn(out.squeeze(), label.squeeze()) + l1_regularization
            loss.backward()
            optimizer.step()
            iteration_loss += loss.item()

        layer1weights.append(model.layers[1].weight.cpu().detach())
        layer2weights.append(model.layers[3].weight[0].cpu().detach())
        train_loss = iteration_loss/len(train_loader)
        trainlosses.append(train_loss)
        print(f'Epoch {epoch + 1}/{epochs}, ' \
                  f'Train Loss: {train_loss:.6f}, ' \
                  f'Test Loss: {compute_test_accuracy(model, validation_loader, loss_fn, device):.6f}')


print("training started", flush=True)
train(model, train_loader, validation_loader, epochs, device)
results = model(x_train_orig).cpu().detach()
# print(results)

bias = model.layers[3].bias
print("Bias:")
print(bias)

# Plotting
figure, axis = pyplot.subplots(2, 2)
axis[0, 0].plot(x_train_orig.cpu().detach(), results, label="prediction")
axis[0, 0].plot(x_train_orig.cpu().detach(), y_train_orig, label="ground truth")
axis[0, 0].legend()
axis[0, 0].set_title("Prediction")

axis[0, 1].plot(trainlosses, label="training loss")
axis[0, 1].legend()
axis[0, 1].set_title("Prediction")


exponentsbyepoch = []
for i in range(0, nrmonomials):
    exponents = []
    for entry in layer1weights:
        exponents.append(entry[i])
    exponentsbyepoch.append(exponents)

coeffsbyepoch = []
for i in range(0, nrmonomials):
    coeffs = []
    for entry in layer2weights:
        coeffs.append(entry[i])
    coeffsbyepoch.append(coeffs)

for i in range(0, nrmonomials):
    axis[1, 0].plot(exponentsbyepoch[i], label=("exponent " + str(i)))
axis[1, 0].legend()
axis[1, 0].set_title("Exponents")

for i in range(0, nrmonomials):
    axis[1, 1].plot(coeffsbyepoch[i], label=("coefficient " + str(i)))
axis[1, 1].legend()
axis[1, 1].set_title("Coefficients")

pyplot.show()



# Explainability: Check with large nr of polynomials plus noise, does it still give approx the right polynomial?
# Data set size: Try with very small dataset as input
# Input domain: What happens when we have negative inputs, do we still learn?
# Experiment with some real datasets. Performance?

# My part:
# 1) check results at high epoch numbers. convergence to actual polynomial?
# 2) Experiment with what happens when we have negative inputs, and also check real datasets, using different numbers of monomials (e.g. housing prices).
