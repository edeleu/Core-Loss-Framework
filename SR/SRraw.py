from julia import Julia

julia = Julia(compiled_modules=False, threads='auto') # optimize = "3"
from julia import Main
from julia.tools import redirect_output_streams

redirect_output_streams()

import pysr

# We don't precompile in colab because compiled modules are incompatible static Python libraries:
pysr.install(precompile=False)

import sympy
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

# Dataset
np.random.seed(0)
X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2

default_pysr_params = dict(
    populations=30,
    procs=4,
    model_selection="best",
)

# Learn equations
model = PySRRegressor(
    niterations=30, # have to restart Jupyter if it takes too long
    binary_operators=["plus", "mult"],
    unary_operators=["cos", "exp", "sin"],
    **default_pysr_params
)

model.fit(X, y)

model.sympy()
# model.equations_.query("loss < 0.1").equation

