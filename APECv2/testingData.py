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

#DEFINE CONSTANTS
# Reproducibility

random.seed(72323)
np.random.seed(72323)
torch.manual_seed(72323)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# substitute for torch.use_deterministic_algorithms(True) in future usage

torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose=True

# Hyperparameters
NUM_EPOCH = 638 #200
BATCH_SIZE = 384
LR_INI = 0.001374771109331439 #0.0026
MIN_LR = 0.0004017121475967859

MUNOT = 1.25663706212e-6

# Set # of epochs to discard due to warmup 
DISCARD = 10

#Materials

#Training Materials 
#materials: material name : [filepath,ID, Bsat, freqPeak, mu_i]
materials = {
    "N87" : ["/scratch/gpfs/ed5754/N87/N87_R34.0X20.5X12.5_Data5_Seq2Scalar_Downsampled_FullFixed.json", 0, 490, 900,2200]
}


def get_datasets(materials):
    datasets = []

    for M, info in materials.items():
        datasets.append(load_dataset(info))

    return torch.utils.data.ConcatDataset(datasets)

def load_dataset(info, data_length=128):
    path = info[0]
    label = info[1]
    bSaturation = info[2]/1000
    freqPeak = info[3] * 1000
    mui = info[4] #initial permeability

    with open(path,'r') as load_f:
        DATA = json.load(load_f)

    B = DATA['B_Field']
    B = np.array(B)
    Freq = DATA['Frequency'] 
    Freq = np.array(Freq) / freqPeak
    #Freq = np.log10(Freq)
    Temp = DATA['Temperature']
    Temp = np.array(Temp)      
    Power = DATA['Volumetric_Loss']     
    Power = np.log10(Power)
    Hdc = DATA['Hdc']
    Hdc = np.array(Hdc)  
    Bdc = MUNOT*mui*Hdc
    
    # Format data into tensors
    Freq = Freq.reshape((-1,1))
    Temp = Temp.reshape((-1,1))
    Hdc = Hdc.reshape((-1,1))
    Bdc = Bdc.reshape((-1,1))
    print(Hdc)
    ("hdc now Bdc yea")
    print(Bdc)

    Power = Power.reshape((-1,1))

    # Combine the T/Hdc scalars into one tensor with two columns (LEN x 2)
    combinedInput = np.concatenate((Temp,Hdc),axis=1)
    parameterTensor = torch.from_numpy(combinedInput).float().view(-1, 2)

    # hold frequency separately to be passed thru projection network
    freqTensor = torch.from_numpy(Freq).float().view(-1,1)

    #Reshape Bdc to N*1*1 then to N*128*1, with Hdc values repeated along the 128 dimension
    BdcTensor = torch.from_numpy(Bdc).float().view(-1,1,1).expand(-1, data_length, -1)

    in_B = torch.from_numpy(B).float().view(-1, data_length, 1)
    # print(in_B.numpy())
    # print("that was in_B")

    #Add BdcTensor to the in_B: for each sequence, a single Bdc value will be added to all points
    in_B_adjusted = torch.add(in_B, BdcTensor)
    # print(in_B_adjusted.numpy())
    # print("now i added BDc")
    in_B_adjusted = torch.div(in_B_adjusted, bSaturation)
    # print(in_B_adjusted.numpy())
    # print("now i divided by bSat")

    #np.savetxt("EDWARD.csv", in_B_adjusted.view(-1,128).numpy(), delimiter=",")
    
    out = torch.from_numpy(Power).float().view(-1,1)

    # put a label with each datapoint, corresponding to the material ID
    labels = torch.full((out.size()), label)

    return torch.utils.data.TensorDataset(in_B_adjusted, freqTensor, parameterTensor, labels, out)

get_datasets(materials)