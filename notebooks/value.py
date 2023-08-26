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

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# ayyygrad won't be in the path by default
import os, sys

if ".." not in sys.path:
    sys.path.insert(0, "..")

# %%
# EXPLANATORY TEXT GOES HERE 
# 
# Note that for the graphviz stuff to work you need to have graphviz installed on your system. 
# For Debian/Ubuntu this is probably something like apt-get install graphviz,
# For Fedora/Redhat its probably dnf install graphviz
# For Arch its pacman -Sy graphviz
#
#
# ...... I use Arch btw

# %%
# This is the network from the Micrograd video 
from ayyygrad.value import Value

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

e = a * b
e.label = "e"

d = e + c
d.label = "d"

f = Value(-2.0, label="f")

L = d * f
L.label = "L"

# %%
# Lets make a graph of L
from ayyygrad.viz import draw

draw(L)

# %%
# If we call backward() on L then the graph should update
L.backward()

draw(L)

# %%
# Try another network

x = Value(-2.0, label="x")
y = Value(3.0, label="y")

xx = x * y; xx.label = "xx"
yy = x + y; yy.label = "yy"

z = xx * yy
z.backward()

draw(z)

# %%
