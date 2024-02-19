
import matplotlib.pyplot as plt
import numpy as np
import torch
from NeuralNetwork_arch import LinearRegressionModel
import torch.nn as nn
import tqdm
import torch.nn.functional as F
import torch.optim as optim

# Setup HW device-agnostic code
if torch.cuda.is_available():
    if args.gpu:
       device = torch.device('cuda:{}'.format(args.gpu)) # NVIDIA GPU
    else:
        device = torch.device('cuda') # NVIDIA GPU
#elif torch.backends.mps_is_available():
#    device = torch.device("mps") # Apple GPU else:
else:
    device = torch.device("cpu")

print(f"Using Device Arithmetic UNIT FROM ISHERIFF Sytem (CPU,GPU,TPU,DPU):{device}")
'''

# load the Raw dataset
X = np.loadtxt('/Users/akram/AKRAM_CODE_FOLDER/LLAMA_LLM_CPP/LLM_EXPLORE/Dataset/pima-indians-diabetes.data.csv', delimiter=',') # split into input (X) and output (y) variables
print(f"SHAPE of Dataset is: {X.shape}")
'''

weight = 0.8
bias = 0.3
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias # labels (want model to learn from data to predict these)

torch.manual_seed(42)
print(f"Generated Dataset is X: {X[:10]},y:{y[:10]}")

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
len(X_train), len(y_train), len(X_test), len(y_test)

model_1 = LinearRegressionModel()
model_1, model_1.state_dict()

# Create loss function
loss_fn = nn.L1Loss()
# Create optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), # optimize newly created model's parameters
lr=0.01)

#torch.manual_seed(42)

epochs = 1000 # number of epochs to run batch_size = 10 # size of each batch batches_per_epoch = len(Xtrain) // batch_size
batch_size = 10
batches_per_epoch = len(X_train) // batch_size

# Set the number of epochs epochs = 1000
# Put data on the available device
# Without this, an error will happen (not all data on target device) X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
# Put model on the available device
# With this, an error will happen (the model is not on target device) model_1 = model_1.to(device)
for epoch in range(epochs): ### Training
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        model_1.train() # train mode is on by default after construction
        # 1. Forward pass
        y_pred = model_1(X_train)
        # 2. Calculate loss
        loss = loss_fn(y_pred, y_train)
        # 3. Zero grad optimizer
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Step the optimizer
        optimizer.step()
        ### Testing
        model_1.eval() # put the model in evaluation mode for testing (inference) # 1. Forward pass

        with torch.inference_mode():
            test_pred = model_1(X_test)
            # 2. Calculate the loss
            test_loss = loss_fn(test_pred, y_test)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")


