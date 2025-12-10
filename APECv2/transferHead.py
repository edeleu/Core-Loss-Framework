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

class MyHead(nn.Module):
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
        x = self.head(x)
        return x

pretrained_dict = torch.load('/scratch/gpfs/ed5754/GeneralFramework466.sd') #pretrained model keys
processed_dict = {}

for k in pretrained_dict.keys():
    if("_orig_mod.heads.0" in k):
        decomposed = k.split(".")
        newKey = ".".join(decomposed[3:])
        print(newKey)
        processed_dict[newKey] = pretrained_dict[k]

myHead = MyHead(13, 11)
myHead.load_state_dict(processed_dict, strict=False)
torch.save(myHead.state_dict(), "/scratch/gpfs/ed5754/ExperimentalHead0.sd")

