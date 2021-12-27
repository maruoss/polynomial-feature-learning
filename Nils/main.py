import sklearn
from sklearn import preprocessing
from sklearn import model_selection
import torch
import scipy
import pandas as pd
from tqdm import tqdm
import numpy as np
import Model



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


def gen_data():
    mylen = 10000
    x_train =  np.reshape(np.arange(1, mylen+1), (-1, 1))
    y_train =  np.reshape(np.arange(2, mylen+2), (-1, 1))
    return x_train, y_train


print("Starting")
x_train, y_train = gen_data()
print("Generated data")
print(x_train[0], y_train[0])

x_train = prepdata(x_train)
print("preprocessing complete")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device is {device}.")


x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=0.1)
print("data splitting complete")

x_train = torch.tensor(x_train, device=device, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, device=device, dtype=torch.float).to(device)
x_val = torch.tensor(x_val, device=device, dtype=torch.float).to(device)
y_val = torch.tensor(y_val, device=device, dtype=torch.float).to(device)

print("tensorification complete")
print(x_train)

batch_size = 64
epochs = 100

train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

# filterlen = [50,10,3]
# strides = [5,5,5]
# featuremap_sizes = [8,32,64]
# model = Model.CNN(featuremap_sizes, filterlen, strides, device)

model = Model.MyModel().to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


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
            loss = loss_fn(out.squeeze(), label.squeeze())
            loss.backward()
            optimizer.step()
            iteration_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, ' \
                  f'Train Loss: {iteration_loss / len(train_loader):.3f}, ' \
                  f'Test Loss: {compute_test_accuracy(model, validation_loader, loss_fn, device):.1f}')


print("training started", flush=True)
train(model, train_loader, validation_loader, epochs, device)