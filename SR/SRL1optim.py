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
import time

torch.set_float32_matmul_precision("high")

# Loss function
# L1 Regularization

# Set number of epochs to discard for warmup
DISCARD = 10

# Reproducibility
random.seed(812)
np.random.seed(812)
torch.manual_seed(812)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# This one has an additional FNN layer 

class SRModel(nn.Module):
    def __init__(self, scalar_dim: int, trial):
        super(SRModel, self).__init__()
                
        n_layers = trial.suggest_int("FNN_layers", 1, 5)
        layers = []
        in_f = scalar_dim
        
        for i in range(n_layers):
                out_features = trial.suggest_int("n_units_{}".format(i), 1, 30)
                layers.append(nn.Linear(in_f, out_features))
                layers.append(nn.ReLU())
                in_f = out_features
        
        layers.append(nn.Linear(in_f, 1))
        self.layers = nn.Sequential(*layers)
        
    def compute_l1_loss(self, w):
      return torch.abs(w).sum()    
        
    def forward(self, x):
        output = self.layers(x)
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_dataset(data_length=128):
    # Load .json Files
    with open('/scratch/gpfs/wl2527/N87_R34.0X20.5X12.5_Data5_sine.json','r') as load_f:
        DATA = json.load(load_f)
    
    Freq = DATA['Frequency']
    Freq = np.log10(Freq)  # logarithm, optional
    Temp = DATA['Temperature']
    Temp = np.array(Temp)      
    Hdc = DATA['Hdc']
    Hdc = np.array(Hdc)  
    Flux = DATA['Flux_Density']
    Flux = np.array(Flux)
    
    Power = DATA['Volumetric_Loss']
    Power = np.log10(Power)

    # Format data into tensors
    Freq = Freq.reshape((-1,1))
    Temp = Temp.reshape((-1,1))
    Hdc = Hdc.reshape((-1,1))
    Flux = Flux.reshape((-1,1))
    Power = Power.reshape((-1,1))

    combinedInput = np.concatenate((Freq,Temp,Hdc,Flux),axis=1)

    in_tensors = torch.from_numpy(combinedInput).float().view(-1, 4)


    print(in_tensors.size())

    return torch.utils.data.TensorDataset(in_tensors)


def evaluate(net, data, device):
    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for in_tensors, out_tensor in data:
            output= net(scalar_input=in_tensors.to(device))
            y_pred.append(output)   
            y_meas.append(out_tensor.to(device))

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(data) * 1e5:.5f}")

    yy_pred = (y_pred.cpu().numpy())
    yy_meas = (y_meas.cpu().numpy())
    
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
    
    return np.mean(Error_re) # np.percentile(Error_re, 95)

def main(trial):
    
    print("Main Loop")

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    NUM_EPOCH = trial.suggest_int("epoch", 100, 600)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [128, 1024])
    # DECAY_EPOCH = 150
    # DECAY_RATIO = 0.9
    LR_INI = trial.suggest_float("LR", 0.001, 0.07) 
    LAMBDA = trial.suggest_float("lambda", 0.01, 10)

    # Select GPU as default device
    device = torch.device("cuda")
    
    dataset = load_dataset() 
    
     # Split the dataset
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    full_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    trainData = list(train_loader)
    validData = list(valid_loader)

    # Setup network

    net = SRModel(scalar_dim=4, trial= trial).to(device)
    
    net = torch.compile(net)

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=165, eta_min=0.00053) #0.0005, 160
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1.1, eta_min=0.0006)
    


    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        #optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))

        start_epoch = time.time()

        for in_tensors, out in trainData:
            optimizer.zero_grad()
            output = net(scalar_input=in_tensors.to(device))
            loss = criterion(output, out.to(device))
            
            # Compute L1 loss component
            l1_weight = LAMBDA
            l1_parameters = []
            for parameter in net.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = l1_weight * net.compute_l1_loss(torch.cat(l1_parameters))
      
            # Add L1 loss component
            loss += l1
            
            loss.backward()
 
            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.25)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Record epoch time 
        # torch.cuda.synchronize()
        # end_epoch = time.time()
        # times.append(end_epoch-start_epoch)
        

        if (epoch_i+1)%10 == 0:
            intermediate_value = evaluate(net, validData, device)
            trial.report(intermediate_value, epoch_i)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
        scheduler.step()
        
    
    return evaluate(net, validData, device)                


if __name__ == "__main__":
    
    # Optuna guidelines: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html#rdb
    
    study_name = "FNN_studying_v1"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction='minimize', 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=90),
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    
    study.optimize(main, n_trials=150, gc_after_trial=True)

    trial = study.best_trial
    print(trial)
    
    main()
