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
    def __init__(self, input_size, task):
        super().__init__()
        self.task=task

        # Standard FFN stack with GELU activation. Output is core loss
        self.head = nn.Sequential(
            nn.Linear(input_size, 18),
            nn.GELU(),
            nn.Linear(18, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        #x: input data, material IDs, output data
        #input data comprises of 2 additional inputs as well as processed projector output
        
        #Find where the material_ID of the input data matches the the ID of the specific-network
        s = torch.where(x[1]==self.task)[0]

        #Activate material network for ONLY the indices where the data has matching material ID
        if s.shape[0] > 0:
            x[2][s] = self.head(x[0][s])
        return x
    
NET = MaterialHead(input_size=13, task=1)

summary(NET, [570, 13])