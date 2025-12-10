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
import optuna

#DEFINE CONSTANTS
# Reproducibility

random.seed(72323)
np.random.seed(72323)
torch.manual_seed(72323)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose=True

# Set # of epochs to discard due to warmup 
DISCARD = 10

#Materials

# materials = {
#     "N87" : ["/scratch/gpfs/sw0123/N87_R34.0X20.5X12.5_Data5_phaseOnly.json", 0],
#     "3C90" : ["/scratch/gpfs/sw0123/3C90_TX-25-15-10_Data1_phaseOnly.json", 1],
#     "3C94" : ["/scratch/gpfs/sw0123/3C94_TX-20-10-7_Data1_phaseOnly.json", 2],
# }


materials = {
    "N87" : ["/scratch/gpfs/sw0123/N87/N87_R34.0X20.5X12.5_Data5_Seq2Scalar_Downsampled_FullFixed.json", 0],
    "3C90" : ["/scratch/gpfs/sw0123/3C90/3C90_TX-25-15-10_Data1_Seq2Scalar_Downsampled_FullFixed.json", 1],
    "3C94" : ["/scratch/gpfs/sw0123/3C94/3C94_TX-20-10-7_Data1_Seq2Scalar_Downsampled_FullFixed.json", 2]
}

#"3F4" : ["/scratch/gpfs/sw0123/3F4_E-32-6-20-R_Data1_Seq2Scalar_Downsampled_FullAll.json", 3],
# "N27" : ["/scratch/gpfs/sw0123/N27_R20.0X10.0X7.0_Data1_Seq2Scalar_Downsampled_FullAll.json", 4]

# Select GPU as default device
device = torch.device("cuda")

# Define material head 

class MaterialHead(nn.Module):
    def __init__(self, output_features, trial, task):
        super().__init__()
        self.task=task

        layers = trial.suggest_int("head_layers", 1, 3)
        layer = []
        in_f = output_features

        for i in range(layers):
                out_features = trial.suggest_int("head_units_{}".format(i), 4, 22)
                layer.append(nn.Linear(in_f, out_features))
                layer.append(nn.GELU())
                in_f = out_features

        layer.append(nn.Linear(in_f, 1))
        self.head = nn.Sequential(*layer)

    def forward(self, x):
        s = torch.where(x[1]==self.task)[0]
        if s.shape[0] > 0:
            x[2][s] = self.head(x[0][s])
        return x

# Define model structures and functions

class LSTMNetwork(nn.Module):
    def __init__(self,
                 dim_val :int,
                 output_features :int,
                 projected_variables :int,
                 num_LSTM :int,
                 trial):
        
        super(LSTMNetwork, self).__init__()

        self.lstm = nn.LSTM(1, dim_val, num_layers=num_LSTM, batch_first=True, bidirectional=False)

        proj_layers = trial.suggest_int("proj_layers", 1, 4)
        proj = []
        proj_in_f = dim_val + projected_variables

        for i in range(proj_layers):
                out_features = trial.suggest_int("proj_units_{}".format(i), 8, 38)
                proj.append(nn.Linear(proj_in_f, out_features))
                proj.append(nn.GELU())
                proj_in_f = out_features

        proj.append(nn.Linear(proj_in_f, output_features))
        self.projector = nn.Sequential(*proj)

        self.heads = nn.Sequential(*[MaterialHead(output_features, trial, id) for id in range(3)])

    def forward(self, in_seq: Tensor, vars: Tensor, materials: Tensor, device) -> Tensor:
        batch = in_seq.shape[0]
        x, _ = self.lstm(in_seq)
        LSTMout = x[:, -1, :] # Get last output only (many-to-one)
        # B x 1 x out_features

        # B x 3 + B x out_feat --> B x (3+dim_val) --> B x output_features
        projector_out = self.projector(torch.cat((vars, LSTMout),dim=1))

        output = torch.zeros(batch, 1, device=device)
        _, _, output = self.heads((projector_out, materials, output))
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load the dataset

def get_datasets(materials):
    datasets = []

    for M, info in materials.items():
        datasets.append(load_dataset(info))

    return torch.utils.data.ConcatDataset(datasets)

def load_dataset(info, data_length=128):
    # Do I want to normalize?
    # Log10 B?

    path = info[0]
    label = info[1]

    # Load .json Files
    with open(path,'r') as load_f:
        DATA = json.load(load_f)

    B = DATA['B_Field']
    B = np.array(B)
    Freq = DATA['Frequency']
    Freq = np.log10(Freq)  # logarithm, optional
    Temp = DATA['Temperature']
    Temp = np.array(Temp)      
    Hdc = DATA['Hdc']
    Hdc = np.array(Hdc)  
    Power = DATA['Volumetric_Loss']     
    Power = np.log10(Power)

    # Format data into tensors
    Freq = Freq.reshape((-1,1))
    Temp = Temp.reshape((-1,1))
    Hdc = Hdc.reshape((-1,1))
    Power = Power.reshape((-1,1))

    combinedInput = np.concatenate((Freq,Temp,Hdc),axis=1)

    in_tensors = torch.from_numpy(combinedInput).float().view(-1, 3)
    in_B = torch.from_numpy(B).float().view(-1, data_length, 1)

    out = torch.from_numpy(Power).float().view(-1,1)

    print(in_B.size())
    print(in_tensors.size())

    labels = torch.full((out.size()), label)

    return torch.utils.data.TensorDataset(in_B, in_tensors, labels, out)

def evaluate(net, data, device):
    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for in_B, in_tensors, labels, out in data:
            y_pred.append(net(in_seq=in_B.to(device), vars=in_tensors.to(device), materials=labels.to(device), device=device))
            y_meas.append(out.to(device))

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
    print("95th percent: ", np.percentile(Error_re, 95))
    print("99th percent: ", np.percentile(Error_re, 99))

    return np.percentile(Error_re, 95)


# Config the model training

def main(trial):
    print("Main loop entered!")

    # Hyperparameters
    NUM_EPOCH = trial.suggest_int("epoch", 70, 700)
    BATCH_SIZE = trial.suggest_int("batch", 256, 1024, 128)
    LR_INI = trial.suggest_float("LR", 0.0009, 0.008)
    MIN_LR = trial.suggest_float("MIN_LR", 0.00006, 0.0006)

    # Load dataset
    dataset = get_datasets(materials)

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    trainData = list(train_loader)
    validData = list(valid_loader)

    # Setup network

    net = LSTMNetwork(dim_val=trial.suggest_int("dim_val", 8, 40), 
                      output_features=trial.suggest_int("out_f", 4, 16), 
                      projected_variables=3, 
                      num_LSTM=trial.suggest_int("LSTMs", 1, 3), 
                      trial=trial).to(device)
    
    net = torch.compile(net)
    # modes: default, reduce-overhead, max-autotune

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=MIN_LR) #0.0005, 160

    # Create list to store epoch times

    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        #optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))

        for in_B, in_tensors, labels, out in trainData:
            optimizer.zero_grad()
            output = net(in_seq=in_B.to(device), vars=in_tensors.to(device), materials=labels.to(device), device=device)
            loss = criterion(output, out.to(device))
            loss.backward()
 
            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.25)
            optimizer.step()
            epoch_train_loss += loss.item()

        if (epoch_i+1)%30 == 0:
            intermediate_value = evaluate(net, validData, device)
            trial.report(intermediate_value, epoch_i)
            # Handle pruning based on the intermediate value.

            if trial.should_prune():
                raise optuna.TrialPruned()
        
        scheduler.step()

    return evaluate(net, validData, device)

if __name__ == "__main__":

    study_name = "FixedLSTM"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction='minimize', 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=90),
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    
    study.optimize(main, n_trials=320, gc_after_trial=True)

    trial = study.best_trial
    print(trial)

    main()
