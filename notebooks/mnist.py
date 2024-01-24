# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MNIST
# Trying to do MNIST from scratch

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from tqdm import trange

# %%
# Try to do mnist from scratch
import numpy as np

# Lifted from geohot stream
def fetch_mnist(url: str) -> np.ndarray:
    import requests, os, gzip, hashlib
    path = os.path.join("/tmp", hashlib.md5(url.encode("utf-8")).hexdigest())
    if os.path.isfile(path):
        with open(path, "rb") as fp:
            data = fp.read()
    else:
        with open(path, "wb") as fp:
            data = requests.get(url).content
            fp.write(data)

    return np.frombuffer(gzip.decompress(data), dtype=np.uint8)

X_train = fetch_mnist("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
Y_train = fetch_mnist("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch_mnist("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch_mnist("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# %%
# Reshape data - TODO: show something about decoding the data format...?
from typing import List

n_dims = X_train[3]

def u8_array_to_int(u8_array: List[int]) -> int:
    return sum([max(256 * idx, 1) * elem for idx, elem in enumerate(reversed(u8_array))])


print(X_train[:0x10])    # X_train header info

dim0_size = u8_array_to_int(X_train[4:8])
dim1_size = u8_array_to_int(X_train[8:12])
dim2_size = u8_array_to_int(X_train[12:16])

#234 * 96
[dim0_size, dim1_size, dim2_size]

# %%
# Lets get the data again but in a useful format
X_train = fetch_mnist("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))

# %%
# What are the shapes of all the datasets?

print(f"X_train.shape: {X_train.shape}")
print(f"Y_train.shape: {Y_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"Y_test.shape: {Y_test.shape}")

# %%
X_train.shape
#X_train.reshape(-1, 28*28).shape

# %%
import matplotlib.pyplot as plt

print(X_train[0].shape)
plt.imshow(X_train[0])

# %%
# Add a classifier 

import torch.nn as nn
from torch import Tensor

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, 10)

    def forward(self, X: Tensor) -> Tensor:
        X = self.l1(X)
        X = self.relu(X)
        X = self.l2(X)
        
        return X

model = TestNet()

# %%
# Train it 

import torch

batch_size = 32
max_iter = 100

loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())


for iter in (t := trange(max_iter)):
    samp = np.random.randint(0, X_train.shape[0], size=(batch_size))
    X = torch.tensor(X_train[samp].reshape((-1, 28 * 28))).float()
    Y = torch.tensor(Y_train[samp]).long()
    optim.zero_grad()
    out = model(X)
    cat = torch.argmax(out, dim=1)
    acc = (cat == Y).float().mean()
    loss = loss_func(out, Y)
    loss.backward()
    optim.step()
    t.set_description(f"Loss: {loss.item()}, Acc: {acc.item()}")

# %%
