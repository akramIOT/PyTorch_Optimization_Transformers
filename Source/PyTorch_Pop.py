# -*- coding: utf-8 -*-
#author: (AKRAM SHERIFF) on 17th,June 2023
##  SOLVING  REGRESSION PROBLEM  WITH PYTORCH  FRAMEWORK.
"""
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim

# load the Raw dataset
dataset = np.loadtxt('/Users/akram/AKRAM_CODE_FOLDER/LLAMA_LLM_CPP/LLM_EXPLORE/Dataset/pima-indians-diabetes.data.csv', delimiter=',') # split into input (X) and output (y) variables
print(f"SHAPE of Dataset is: {dataset.shape}")

X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
# split the dataset into training and test sets
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
print(model)
# loss function and optimizer
loss_fn = nn.BCELoss() # binary cross entropy optimizer = optim.Adam(model.parameters(), lr=0.0001)
n_epochs = 50 # number of epochs to run batch_size = 10 # size of each batch batches_per_epoch = len(Xtrain) // batch_size
batch_size = 10
batches_per_epoch = len(Xtrain) // batch_size

# define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# collect statistics
train_loss = []
train_acc = []
test_acc = []

for epoch in range(n_epochs):
     with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
          bar.set_description(f"Epoch {epoch}")
          for i in bar:
          # take a batch
             start = i * batch_size
             Xbatch = Xtrain[start:start+batch_size]
             ybatch = ytrain[start:start+batch_size]
             # forward pass
             y_pred = model(Xbatch)
             loss = loss_fn(y_pred, ybatch)
             acc = (y_pred.round() == ybatch).float().mean() # store metrics
             train_loss.append(float(loss))
             train_acc.append(float(acc))
             # backward pass
             optimizer.zero_grad()
             loss.backward()
             # update weights
             optimizer.step()
             # print progress
             bar.set_postfix(
                 loss=float(loss),
          acc=f"{float(acc)*100:.2f}%" )

# evaluate model at end of epoch
y_pred = model(Xtest)
acc = (y_pred.round() == ytest).float().mean()
test_acc.append(float(acc))
print(f"End of {epoch}, accuracy {acc}")

 # Plot the loss metrics
plt.plot(train_loss)
plt.xlabel("steps")
plt.ylabel("loss")
plt.ylim(0)
plt.show()

 # plot the accuracy metrics - Matplotlib
avg_train_acc = []
for i in range(n_epochs):
     start = i * batch_size
     average = sum(train_acc[start:start+batches_per_epoch]) / batches_per_epoch
     avg_train_acc.append(average)
plt.plot(avg_train_acc, label="train")
plt.plot(test_acc, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0)
plt.show()

