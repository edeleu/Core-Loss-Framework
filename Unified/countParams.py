# Import necessary packages

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
from torchinfo import summary

class MaterialHead(nn.Module):
    def __init__(self, output_features, task):
        super().__init__()
        self.task=task

        self.head = self.head = nn.Sequential(
            nn.Linear(output_features, 15),
            nn.GELU(),
            nn.Linear(15, 18),
            nn.GELU(),
            nn.Linear(18, 1)
        )
    def forward(self, x):
        s = torch.where(x[1]==self.task)[0]
        if s.shape[0] > 0:
            x[2][s] = self.head(x[0][s])
        return x
    
NET = MaterialHead(output_features=14, task=1)

summary(NET, [57, 14])