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
    def __init__(self):
        super(Net, self).__init__()
        # Define a fully connected layers model with three inputs (frequency, flux density, duty ratio) and one output (power loss).
        self.layers = nn.Sequential(
            nn.Linear(5, 21),
            nn.GELU(),
            nn.Linear(21, 12),
            nn.GELU(),
            nn.Linear(12, 5),
            nn.GELU(),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        return self.layers(x)
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# # Step 2: Load the Dataset
# In this part, we load and pre-process the dataset for the network training and testing. In this demo, a small dataset containing triangular waveforms measured with N87 ferrite material under different frequency, flux density, and duty ratio is used, which can be downloaded from the [MagNet GitHub](https://github.com/PrincetonUniversity/Magnet) repository under "tutorial". 


# Load the dataset

def load_datasets():
    
    pretrain_sets = []
    for M, path in materials.items():
        if M != test_material:
            pretrain_sets.append(get_dataset(path))

    pretrain_dataset = torch.utils.data.ConcatDataset(pretrain_sets)

    test_set = get_dataset(materials[test_material])

    return pretrain_dataset, test_set

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

def pretrain():
    # Load dataset
    train, test = load_datasets()

    # Split the dataset
    train_size = int(0.6 * len(train))
    valid_size = int(0.2 * len(train))
    test_size = len(train) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(train, [train_size, valid_size, test_size])

    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    trainLoop = list(train_loader)

    # Setup network
    net = Net().double().to(device)
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

        # Compute Validation Loss
        if (epoch_i+1)%100 == 0:
          with torch.no_grad():
            epoch_valid_loss = 0
            for inputs, labels in valid_loader:
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels.to(device))

                epoch_valid_loss += loss.item()
          
          print(f"Epoch {epoch_i+1:2d} "
              f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
              f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")
        
    # Save the model parameters
    torch.save(net.state_dict(), "pretrainedMulti.sd")
    print("Training finished! Model is saved!")

    print("testing pretrained model with combined dataset")
    # Evaluation
    evaluate(net, test_loader)

    print("test model with single material, 3C94, no retrain")
    seenLoader = torch.utils.data.DataLoader(get_dataset(materials["3C94"]), batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    evaluate(net, seenLoader)

    print("test model with unseen material, N87, no retrain")
    unseenLoader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    evaluate(net, unseenLoader)
    
    print("retraining now!")
    retrain(net.state_dict(), test)

def retrain(trial):

    RETRAIN_EPOCH = trial.suggest_int("epoc", 20, 100)
    LR_INI = trial.suggest_float("lr", 1e-6, 1e-2)
    BATCH_SIZE = NUM_RETRAIN / 4

    test = get_dataset(materials[test_material])
    netSD = torch.load('/scratch/gpfs/sw0123/pretrainedMulti.sd')

    network = Net().double().to(device)
    network = torch.compile(network)

    network.load_state_dict(netSD, strict=True)

    train_dataset, test_dataset, _ = torch.utils.data.random_split(test, [NUM_RETRAIN, len(test) -
                                                        NUM_RETRAIN, 0])

    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    trainLoop = list(train_loader)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=LR_INI) 

    # Train the network
    for epoch_i in range(RETRAIN_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        network.train()
        optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))

        for inputs, labels in trainLoop:
            optimizer.zero_grad()
            outputs = network(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
          
        #print(f"Epoch {epoch_i+1:2d} "
        #     f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} ")
    
    #print("finally, test on final material test data")

    # Evaluation
    #evaluate(network, test_loader)
    network.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            y_pred.append(network(inputs.to(device)))
            y_meas.append(labels.to(device))

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(test_loader) * 1e5:.5f}")

    yy_pred = 10**(y_pred.cpu().numpy())
    yy_meas = 10**(y_meas.cpu().numpy())
    
    # Relative Error
    Error_re = abs(yy_pred-yy_meas)/abs(yy_meas)*100
    Error_re_avg = np.mean(Error_re)
    Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    Error_re_max = np.max(Error_re)

    return Error_re_max


def evaluate(net, data):
    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in data:
            y_pred.append(net(inputs.to(device)))
            y_meas.append(labels.to(device))

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(data) * 1e5:.5f}")

    yy_pred = 10**(y_pred.cpu().numpy())
    yy_meas = 10**(y_meas.cpu().numpy())
    
    # Relative Error
    Error_re = abs(yy_pred-yy_meas)/abs(yy_meas)*100
    Error_re_avg = np.mean(Error_re)
    Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    Error_re_max = np.max(Error_re)
    print(f"Relative Error: {Error_re_avg:.8f}")
    print(f"RMS Error: {Error_re_rms:.8f}")
    print(f"MAX Error: {Error_re_max:.8f}")

def train_single():
    data = get_dataset(materials[test_material])

    # Split the dataset
    train_size = int(0.6 * len(data))
    valid_size = int(0.2 * len(data))
    test_size = len(data) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, valid_size, test_size])

    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    trainLoop = list(train_loader)

    # Setup network
    net = Net().double().to(device)
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

        # Compute Validation Loss
        if (epoch_i+1)%100 == 0:
          with torch.no_grad():
            epoch_valid_loss = 0
            for inputs, labels in valid_loader:
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels.to(device))

                epoch_valid_loss += loss.item()
          
          print(f"Epoch {epoch_i+1:2d} "
              f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
              f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")

        
    # Save the model parameters
    print("Training finished! Model is saved!")

    # Evaluation
    evaluate(net, test_loader)

    
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
    NUM_RETRAIN = 100
    RETRAIN_EPOCH = 80


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


    #train_single()
    #pretrain()

    # state_dict = torch.load('/scratch/gpfs/sw0123/pretrainedMulti.sd')
    # test = get_dataset(materials[test_material])
    # retrain(state_dict, test)
    # study = optuna.create_study(direction='minimize')
    # study.optimize(retrain, n_trials=100)
    # optimal: [I 2023-07-07 11:35:53,523] Trial 20 finished with value: 56.801754267404704 and parameters: {'epoc': 69, 'lr': 0.006230707568339242}.

    # code is currently set up for optimizing retrain step. to return to normal pipeline slight modifications required to get rid of optuna and pass from pretrain to retrain