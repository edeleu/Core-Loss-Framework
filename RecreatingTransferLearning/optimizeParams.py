# # Part 1: Network Training 
# This tutorial demonstrates how to train the FNN-based model for the core loss prediction. The network model will be trained based on Dataset_tri.json and saved as a state dictionary (.sd) file.
# 


# # Step 0: Import Packages
# In this demo, the neural network is synthesized using the PyTorch framework. Please install PyTorch according to the [official guidance](https://pytorch.org/get-started/locally/) , then import PyTorch and other dependent modules.

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
import optuna

# # Step 1: Define Network Structure
# In this part, we define the structure of the feedforward neural network. Refer to the [PyTorch document](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) for more details.


# Define model structures and functions

class Net(nn.Module):
    def __init__(self, trial):
        super(Net, self).__init__()

        activ = trial.suggest_categorical("activ", ["ReLU", "GELU"])

        if activ == "ReLU":        
            n_layers = trial.suggest_int("n_layers", 2, 5)
            layers = []

            in_features = 5
            for i in range(n_layers):
                out_features = trial.suggest_int("n_units_l{}".format(i), 5, 25)
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.ReLU())
                in_features = out_features

            layers.append(nn.Linear(in_features, 1))

            self.layers = nn.Sequential(*layers)

        else:
            n_layers = trial.suggest_int("n_layers", 2, 5)
            layers = []

            in_features = 5
            for i in range(n_layers):
                out_features = trial.suggest_int("n_units_l{}".format(i), 5, 25)
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.GELU())
                in_features = out_features

            layers.append(nn.Linear(in_features, 1))

            self.layers = nn.Sequential(*layers)
            

    def forward(self, x):
        return self.layers(x)
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# # Step 2: Load the Dataset
# In this part, we load and pre-process the dataset for the network training and testing. In this demo, a small dataset containing triangular waveforms measured with N87 ferrite material under different frequency, flux density, and duty ratio is used, which can be downloaded from the [MagNet GitHub](https://github.com/PrincetonUniversity/Magnet) repository under "tutorial". 


# Load the dataset

def get_dataset(path):
    # Load .json Files
    with open(path,'r') as load_f:
        DATA = json.load(load_f)
        
    Freq = DATA['Frequency']
    Tempe = DATA['Temperature']
    Flux = DATA['Flux_Density']
    Duty = DATA['Duty_P']
    Power = DATA['Volumetric_Loss']
    Hbias = DATA['Hdc']

    # Compute labels
    # There's approximalely an exponential relationship between Loss-Freq and Loss-Flux. 
    # Using logarithm may help to improve the training.
    Freq = np.log10(Freq)
    Flux = np.log10(Flux)
    Duty = np.array(Duty)
    Power = np.log10(Power)
    Temps = np.array(Tempe)
    Hdc = np.array(Hbias)
    
    # Reshape data
    Freq = Freq.reshape((-1,1))
    Flux = Flux.reshape((-1,1))
    Duty = Duty.reshape((-1,1))
    Temps = Temps.reshape((-1,1))
    Hdc = Hdc.reshape((-1,1))

    temp = np.concatenate((Freq,Flux,Duty,Hdc,Temps),axis=1)
    
    in_tensors = torch.from_numpy(temp).view(-1, 5)
    out_tensors = torch.from_numpy(Power).view(-1, 1)

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)


# # Step 3: Training and Testing the Model
# In this part, we program the training and testing procedure of the network model. The loaded dataset is randomly split into training set, validation set, and test set. The output of the training is the state dictionary file (.sd) containing all the trained parameter values.


# Config the model training    

def objective(trial):
    
    data = get_dataset(materials[test_material])
    #BATCH_SIZE = trial.suggest_int("batch", 128, 1024, step=128)
    #NUM_EPOCH = trial.suggest_int("epoch", 500, 1500, step=50)

    # Split the dataset
    train_size = int(0.7 * len(data))
    test_size = len(data) - train_size
    train_dataset, _, test_dataset = torch.utils.data.random_split(data, [train_size, 0, test_size])

    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    trainLoop = list(train_loader)
    testLoop = list(test_loader)

    # Setup network
    net = Net(trial).double().to(device)
    net = torch.compile(net)

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 

    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))

        for inputs, labels in trainLoop:
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in testLoop:
            y_pred.append(net(inputs.to(device)))
            y_meas.append(labels.to(device))

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(test_dataset) * 1e5:.5f}")

    yy_pred = 10**(y_pred.cpu().numpy())
    yy_meas = 10**(y_meas.cpu().numpy())
    
    # Relative Error
    Error_re = abs(yy_pred-yy_meas)/abs(yy_meas)*100
    Error_re_avg = np.mean(Error_re)
    Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    Error_re_max = np.max(Error_re)

    return Error_re_avg, Error_re_max, count_parameters(net)

    
if __name__ == "__main__":
    
    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    NUM_EPOCH = 1200
    BATCH_SIZE = 384
    DECAY_EPOCH = 100
    DECAY_RATIO = 0.5
    LR_INI = 0.02


    # Select GPU as default device
    device = torch.device("cuda")

    test_material = "N87"
    materials = {
                '3C90': "TransferData/3C90_TX-25-15-10_Data1_Scalar2Scalar_Triangular.json", 
                '3C94': "TransferData/3C94_TX-20-10-7_Data1_Scalar2Scalar_Triangular.json", 
                '3F4': "TransferData/3F4_E-32-6-20-R_Data1_Scalar2Scalar_Triangular.json", 
                'N49': "TransferData/N49_R16.0X9.6X6.3_Data1_Scalar2Scalar_Triangular.json",
                'N87': "TransferData/N87_R34.0X20.5X12.5_Data5_Scalar2Scalar_Triangular.json"
                }

    study = optuna.create_study(directions=['minimize', 'minimize', 'minimize'])
    study.optimize(objective, n_trials=1000)

    #trial = study.best_trial  
    #print(trial)

    print(study.best_trials)  