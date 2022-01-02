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

print('cusa available: ', torch.cuda.is_available())

# use either cpu or gpu
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


#use minmax scaler instead
def prepdata(x):
    x = pd.read_csv(x, sep="\s+", header=None)
    # x = pd.DataFrame(x)
    z_scores = x.apply(scipy.stats.zscore, nan_policy='omit')
    # print(z_scores)
    # print(x.size)
    # x.where(abs(z_scores) < 3, inplace=True)
    # print(x.isna().any(axis=0))
    x = x.fillna(x.mean())
    x = x.to_numpy()
    scaler = MinMaxScaler((1, math.exp(1)))
    scaler.fit(x)
    x = scaler.transform(x)
    return x


def split_xy(x, y_list):
    y = []
    y_list.sort(reverse=True)
    for i in y_list:
        y.append(x[:,i])
        # housing_normed = np.delete(housing_normed, 4, 1)
        x = np.delete(x, i, 1)
    y = np.transpose(y)
    return torch.FloatTensor(x).to(device), torch.FloatTensor(y).to(device)


x = prepdata('C:\\Users\\Ahmet\\ETH_Master\\DeepLearning\\feature_learning_project\\Datasets\\housing.data')
x, y = split_xy(x, [13])  # [13, 4] for evaluating both

def train_and_test(X, y, nnet, mon_dim, b1, b2, bs, lr, max_epochs=1000, nsplits=5, printout=100, show_pred=False):
    # X, y = make_data(n, f_num)
    strt=time.time()
    input_dim = X.size(dim=1)
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
        # prev_loss = inf
        if nnet == 'poly':
            net = Poly_Net(input_dim, mon_dim, 2, b1, b2)
        else:
            net = Relu_Net(input_dim, mon_dim, 2, b1, b2)
        net.to(device)
        criterion = nn.MSELoss()
        # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.005)
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
                # with torch.no_grad():
                #     net.lin_hidden.weight.clamp_(0.)
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
    # X, y = make_data(n, f_num)
    input_dim = X.size(dim=1)
    kf = KFold(n_splits=nsplits, shuffle=True)
    # loss list for every k fold
    val_loss = []
    for j, indices in enumerate(kf.split(X)):
        print('-- split: ', j+1)
        train_index, test_index = indices
        # print(type(train_index), type(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        loss = mean_squared_error(y_test, y_pred)
        val_loss.append(loss)
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


poly_k_loss, poly_v_loss = train_and_test(X=x, y=y, nnet='poly', mon_dim=100, b1=False, b2=False, bs=8, lr=0.0005,
                                          max_epochs=5000, nsplits=5, printout=500)

k_plots([poly_k_loss], ['blue'])
