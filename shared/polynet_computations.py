import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import math
import numpy as np
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import scipy
from model import Poly_Net, Relu_Net

print('cuda available: ', torch.cuda.is_available())

# use either cpu or gpu
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


def read_data(path, y_list, s="\s+"):
    x = pd.read_csv(path, sep=s, header=None)
    # x = pd.DataFrame(x)
    x = x.to_numpy()
    y = []
    y_list.sort(reverse=True)
    for i in y_list:
        y.append(x[:, i])
        # housing_normed = np.delete(housing_normed, 4, 1)
        x = np.delete(x, i, 1)
    y = np.transpose(y)
    return x, y


def prep_data(x_train, x_test, y_train, y_test):
    # prep x data
    x_train = pd.DataFrame(data=x_train)
    z_scores = x_train.apply(scipy.stats.zscore, nan_policy='omit')
    x_train.where(abs(z_scores) < 3, inplace=True)
    x_train = x_train.fillna(x_train.mean())
    x_train = x_train.to_numpy()
    scaler = MinMaxScaler((1, math.exp(1)))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # eliminate negative values
    x_test[x_test < 1] = 1
    x_test[x_test > math.exp(1)] = math.exp(1)
    # prep y data
    y_train = pd.DataFrame(data=y_train)
    z_scores = y_train.apply(scipy.stats.zscore, nan_policy='omit')
    y_train.where(abs(z_scores) < 3, inplace=True)
    y_train = y_train.fillna(y_train.mean())
    y_train = y_train.to_numpy()
    # scaler = MinMaxScaler((1, math.exp(1)))
    scaler.fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    return torch.FloatTensor(x_train).to(device), torch.FloatTensor(x_test).to(device), torch.FloatTensor(y_train).to(
        device), torch.FloatTensor(y_test).to(device)


x, y = read_data('datasets/housing.data', [13, 4])  # [13] for only predicting PRICE and not NOX


def train_and_test(X, y, nnet, mon_dim, b1, b2, bs, lr, max_epochs=1000, nsplits=5, printout=100, show_pred=False):
    # X, y = make_data(n, f_num)
    strt=time.time()
    input_dim = X.shape[1]
    kf = KFold(n_splits=nsplits, shuffle=True)
    # loss list for every k fold
    k_loss = [[] for _ in range(nsplits)]
    val_loss = []
    for j, indices in enumerate(kf.split(X)):
        print('\n============= split '+str(j+1)+' =============\n')
        train_index, test_index = indices
        # print(type(train_index), type(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test, y_train, y_test = prep_data(X_train, X_test, y_train, y_test)
        # prev_loss = inf
        out_dim = y.shape[1]
        if nnet == 'poly':
            net = Poly_Net(input_dim, mon_dim, out_dim, b1, b2)
        else:
            net = Relu_Net(input_dim, mon_dim, out_dim, b1, b2)
        net.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.8)
        # optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.005)
        training_data = DataLoader(list(zip(X_train, y_train)), batch_size=bs, shuffle=True)
        for epoch in range(max_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(training_data):
                # get the inputs; data is a list of [inputs, labels]
                inputs, result = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs.to(device))
                loss = criterion(outputs, result)
                loss.backward()
                # nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_norm)
                optimizer.step()
                # clamp zero vals
                if isinstance(net, Poly_Net):
                    with torch.no_grad():
                        net.lin_hidden.weight.clamp_(0.)
                # print statistics
                running_loss += loss.item()
            k_loss[j].append(running_loss/len(training_data))
            if epoch % printout == printout-1:
                print('epoch: ', epoch+1, '; loss: ', running_loss/len(training_data))
        # validation set
        out = net(X_test)
        loss = criterion(out, y_test)
        print('\nvalidation loss: ', loss.item())
        val_loss.append(loss.item())
    print('\n\n--------------\n\n')
    print('average loss: ', sum(val_loss)/nsplits)
    print('cross validation train and test took: '+str(time.time()-strt))
    return k_loss, val_loss


def train_and_test_linreg(X, y, nsplits=5):
    input_dim = X.shape[1]
    kf = KFold(n_splits=nsplits, shuffle=True)
    # loss list for every k fold
    val_loss = []
    for j, indices in enumerate(kf.split(X)):
        print('-- split: ', j+1)
        train_index, test_index = indices
        # print(type(train_index), type(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test, y_train, y_test = prep_data(X_train, X_test, y_train, y_test)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        loss = mean_squared_error(y_test, y_pred)
        val_loss.append(loss)
        print('val loss: ', loss)
    print('average loss: ', sum(val_loss)/nsplits)
    return val_loss


# plot the loss along learning epochs for all folds and finally compute the average learning loss across the folds with 'avgs'
def k_plots(k_losses, colors):
    plt.yscale('log')
    for i, kl in enumerate(k_losses):
        assert len(kl) > 0
        avgs = [0 for _ in kl[0]]
        for _, k in enumerate(kl):
            plt.plot(k, alpha=0.3, color=colors[i])
            for j, ki in enumerate(k):
                avgs[j] += ki
        avgs = [x/len(kl) for x in avgs]
        plt.plot(avgs, color=colors[i])
    plt.show()

poly_k_loss, poly_v_loss = train_and_test(X=x, y=y, nnet='poly', mon_dim=100, b1=False, b2=True, bs=8, lr=0.0005, max_epochs=5000, nsplits=5, printout=500)

net_k_loss, net_v_loss = train_and_test(X=x, y=y, nnet='net', mon_dim=100, b1=True, b2=True, bs=8, lr=0.0005, max_epochs=5000, nsplits=5, printout=500)

linreg_v_loss = train_and_test_linreg(X=x, y=y, nsplits=5)

k_plots([poly_k_loss], ['blue'])
