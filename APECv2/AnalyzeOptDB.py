
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import json
import math
import time
#import nvtx
import torch._dynamo as dynamo
import csv
import optuna

study_name = "Full78TL"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
#fig = optuna.visualization.plot_intermediate_values(study)
#fig = optuna.visualization.plot_param_importances(study)
fig = optuna.visualization.plot_optimization_history(study)
#fig = optuna.visualization.plot_contour(study, params=["out_f", "head_layers"])
#fig = optuna.visualization.plot_slice(study, params=["out_f"])

fig.show()

print(study.best_trial.params)
print(study.best_value)