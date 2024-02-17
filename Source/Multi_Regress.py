# -*- coding: utf-8 -*-
#author: isheriff@cisco.com (AKRAM SHERIFF) - 17th,November' 2023 
""" 
PyTorch Framework Gradients Calculation functionality working exploration 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np

inputs = np.array([[73, 67, 43], [91, 88, 64],
[87, 134, 58],[102, 43, 37],
[69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], [81, 101],
[119, 133],
[22, 37],
[103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)

w = torch.randn(2,3, requires_grad= True) 
b = torch.randn(2, requires_grad= True) 
print(w,'\n',b)

def model(x):
  return x @ w.t() + b

preds = model(inputs)
print(targets)

def mse(t1,t2):
    diff = t1-t2
    return torch.sum(diff*diff)/diff.numel()

loss = mse(preds,targets)
print(loss)

loss.backward()
print(w) 
print(w.grad)

with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    
loss = mse(preds, targets) 
print(loss)

w.grad.zero_() 
b.grad.zero_() 
print(w.grad,'\n',b.grad)

preds = model(inputs) 
print(preds)

loss = mse(preds, targets) 
print(loss)

loss.backward() 
print(w.grad) 
print(b.grad)

with torch.no_grad(): 
    w -= w.grad * 1e-5 
    b -= b.grad * 1e-5 
    w.grad.zero_() 
    b.grad.zero_()

print(w)
print(b)

preds = model(inputs) 
loss = mse(preds, targets) 
print(loss)

for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
         w -= w.grad * 1e-5
         b -= b.grad * 1e-5
         w.grad.zero_()
         b.grad.zero_()
         
preds = model(inputs) 
loss = mse(preds, targets) 
print(f"LOSS: {loss} MSE loss \n")
print(f"PREDS: {preds} \n")
print(f"TARGETS: {targets} \n")






















