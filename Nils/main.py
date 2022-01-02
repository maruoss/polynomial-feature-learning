import matplotlib.pyplot
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
import torch
import scipy
import pandas as pd
from tqdm import tqdm
import numpy as np

import Comparisonmodel
import Model
from matplotlib import pyplot
from math import exp


#use minmax scaler instead
def prepdata(x):
    x = pd.DataFrame(x)
    z_scores = x.apply(scipy.stats.zscore, nan_policy='omit')
    x.where(abs(z_scores) < 3, inplace=True)
    x = x.fillna(x.mean())
    x = x.to_numpy()
    scaler = preprocessing.MinMaxScaler((1, exp(1)))
    scaler.fit(x)
    x = scaler.transform(x)
    return x


def compute_test_accuracy(model, val_loader, loss_fn, device):
    accuracy = []
    for batch in tqdm(val_loader, leave=False):
        X, label = batch
        outputs = model(X)
        acc = loss_fn(outputs, torch.unsqueeze(label, 1))
        # print(acc)
        accuracy.append(acc)
    accuracy = torch.tensor(accuracy, device=device, dtype=torch.float).to(device)
    return torch.mean(accuracy)


def gen_func(x):
    y =  5*(x**4) + 2
    return y

def gen_data():
    mylen = 10000
    x_train =  np.reshape(np.arange(0, mylen)/mylen, (-1, 1))
    y_train =  gen_func(x_train)
    return x_train, y_train

def train(model, train_loader, validation_loader, epochs, device, optimizer, loss_fn):
    finalvalloss = 0
    for epoch in range(epochs):
        iteration_loss = 0.
        for batch in tqdm(train_loader, leave=False):
            # print(batch)
            x, label = batch
            #print(batch)
            optimizer.zero_grad()
            out = model(x)
            # print(out.squeeze().shape, label.squeeze().shape)
            #all_linear2_params = torch.cat([x.view(-1) for x in model.lin2.parameters()])
            #all_linear3_params = torch.cat([x.view(-1) for x in model.lin3.parameters()])
            #l1_regularization = (0.00005 * torch.norm(all_linear2_params, 1))+(0.0005 * torch.norm(all_linear3_params, 1))
            loss = loss_fn(out, torch.unsqueeze(label, 1))
            loss.backward()
            optimizer.step()
            model.force_non_negative_exponents_()
            iteration_loss += loss.item()

        #if (epoch % 50) == 0:
            #layer1weights.append(model.layers[1].weight.cpu().detach())
            #layer2weights.append(model.layers[3].weight[0].cpu().detach())
        train_loss = iteration_loss/len(train_loader)

        #trainlosses.append(train_loss)
        valloss = compute_test_accuracy(model, validation_loader, loss_fn, device).cpu().detach()
        finalvalloss = valloss
        #vallosses.append(valloss)
        print(f'Epoch {epoch + 1}/{epochs}, ' \
                  f'Train Loss: {train_loss:.6f}, ' \
                  f'Test Loss: {valloss:.6f}')
    return finalvalloss

def trainrelu(model, train_loader, validation_loader, epochs, device, optimizer, loss_fn):
    finalvalloss = 0
    for epoch in range(epochs):
        iteration_loss = 0.
        for batch in tqdm(train_loader, leave=False):
            # print(batch)
            x, label = batch
            #print(batch)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, torch.unsqueeze(label, 1))
            loss.backward()
            optimizer.step()
            iteration_loss += loss.item()
        train_loss = iteration_loss/len(train_loader)
        valloss = compute_test_accuracy(model, validation_loader, loss_fn, device).cpu().detach()
        finalvalloss = valloss
        #vallosses.append(valloss)
        print(f'Epoch {epoch + 1}/{epochs}, ' \
                  f'Train Loss: {train_loss:.6f}, ' \
                  f'Test Loss: {valloss:.6f}')
    return finalvalloss


print("Starting")
#x_train, y_train = gen_data()
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
x_train = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
#print(x_train.shape)
y_train = raw_df.values[1::2, 2]
x_train1, x_test1, y_train1, y_test1 = model_selection.train_test_split(x_train, y_train, test_size=0.1)
print("loaded data")
#print(x_train, y_train)
x_train_orig = x_train1
y_train_orig = y_train1

x_train1 = prepdata(x_train1)
x_test1 = prepdata(x_test1)
print("preprocessing complete")
#print(x_train)

nrmonomials = 5
nrvariables = x_train.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device is {device}.")

kf = model_selection.KFold(n_splits=5)
vallosses = []
vallossesrelu = []
asdf = 0
for train_idx, test_idx in kf.split(x_train1):
    x_train, x_val, y_train, y_val = x_train1[train_idx], x_train1[test_idx], y_train1[train_idx], y_train1[test_idx]
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

    torch.manual_seed(68225)
    vallosses.append([])
    vallossesrelu.append([])
    for nrpolynomials in range(1, 103, 5):
        model = Model.PolynomialNN(13, nrpolynomials, False).to(device)
        with torch.no_grad():
            model.exponents_abs_()
        relunet = Comparisonmodel.ReluNet(13, nrpolynomials).to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.005)

        epochs = 5000

        print("training started", flush=True)
        valloss = train(model, train_loader, validation_loader, epochs, device, optimizer, loss_fn)
        reluloss = trainrelu(relunet, train_loader, validation_loader, epochs, device, optimizer, loss_fn)
        vallosses[asdf].append(valloss)
        vallossesrelu[asdf].append(reluloss)
        np.savetxt("vallosses.csv", vallosses, delimiter=",")
        np.savetxt("vallossesrelu.csv", vallossesrelu, delimiter=",")
    asdf+=1

# print(results)

np.savetxt("vallosses.csv", vallosses, delimiter=",")
np.savetxt("vallossesrelu.csv", vallossesrelu, delimiter=",")
# Plotting
#axis[0, 0].plot(results, label="prediction")
#axis[0, 0].plot(x_train_orig.cpu().detach(), y_train_orig, label="ground truth")
#axis[0, 0].legend()
#axis[0, 0].set_title("Prediction")


dim_axis = np.arange(1, 103, 5)
#axis[0, 1].plot(trainlosses,  label="training loss")
#axis[0, 1].set_yscale("log")
#axis[0, 1].legend()
#axis[0, 1].set_title("Training")

pyplot.plot(dim_axis, np.mean(vallosses, axis=0),  label="PolyNet")
pyplot.plot(dim_axis, np.mean(vallossesrelu, axis=0), label="ReluNet")
pyplot.legend()
pyplot.title("Losses for PolyNet vs ReluNet")
pyplot.xlabel("Number of Internal Dimensions")
pyplot.ylabel("Validation Loss")
pyplot.show()

#print(layer1weights)
