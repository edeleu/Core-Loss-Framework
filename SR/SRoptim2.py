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

# Set number of epochs to discard for warmup
DISCARD = 10

# This one has an additional FNN layer 

class SRModel(nn.Module):
    def __init__(self, hidden_dim, scalar_dim, output_features, num_LSTM: int, trial):
        super(SRModel, self).__init__()
        
        # optim 1
        self.lstm = nn.LSTM(1, hidden_dim, num_layers = num_LSTM, batch_first = True, bidirectional = False)
        
        n_layers = trial.suggest_int("FNN_layers", 3, 15)
        layers = []
        in_f = hidden_dim
        
        for i in range(n_layers):
                out_features = trial.suggest_int("n_units_{}".format(i), 12, 60)
                layers.append(nn.Linear(in_f, out_features))
                layers.append(nn.ReLU())
                in_f = out_features
        
        layers.append(nn.Linear(in_f, output_features))
        self.lstm_out_projector = nn.Sequential(*layers)
        
        # optim 2
        n_layers_final = trial.suggest_int("FNN_final_layers", 1, 10)
        layers_final = []
        in_f_final = output_features + scalar_dim
        
        for i in range(n_layers_final):
                out_features_final = trial.suggest_int("final_fnn_units_{}".format(i), 4, 30)
                layers_final.append(nn.Linear(in_f_final, out_features_final))
                layers_final.append(nn.ReLU())
                in_f_final = out_features_final
        
        layers_final.append(nn.Linear(in_f_final, 1))
        self.final_projector = nn.Sequential(*layers_final)
        
        
    def forward(self, sequence_input, scalar_input):
        x, _ = self.lstm(sequence_input)
        LSTM_out = x[:, -1, :] # many to one
        lstm_out_output = self.lstm_out_projector(LSTM_out)
        output = self.final_projector(torch.cat((scalar_input, lstm_out_output),dim=1))
        return output, lstm_out_output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_dataset(data_length=128):
    # Load .json Files
    with open('/scratch/gpfs/wl2527/N87_R34.0X20.5X12.5_Data5_Seq2Scalar.json','r') as load_f:
        DATA = json.load(load_f)

    Volt = DATA['Voltage']
    Volt = np.array(Volt)
    
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
    in_volt = torch.from_numpy(Volt).float().view(-1, data_length, 1)

    out_tensor = torch.from_numpy(Power).float().view(-1,1)

    print(in_volt.size())
    print(in_tensors.size())

    return torch.utils.data.TensorDataset(in_volt, in_tensors, out_tensor)


def evaluate(net, data, device):
    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for in_volt, in_tensors, out_tensor in data:
            output, _ = net(sequence_input=in_volt.to(device), scalar_input=in_tensors.to(device))
            y_pred.append(output)   
            y_meas.append(out_tensor.to(device))

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(data) * 1e5:.5f}")

    yy_pred = 10 ** (y_pred.cpu().numpy())
    yy_meas = 10 ** (y_meas.cpu().numpy())
    
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
    
    final_outputs = []
    final_hidden_layers = []
    start_time = time.time()

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    NUM_EPOCH = trial.suggest_int("epoch", 100, 1000)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [128, 1024])
    # DECAY_EPOCH = 150
    # DECAY_RATIO = 0.9
    LR_INI = trial.suggest_float("LR", 0.001, 0.07)


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

    net = SRModel(hidden_dim=trial.suggest_int("hidden_dim", 5, 40), scalar_dim=4, output_features= trial.suggest_int("output_features", 20, 40),
                  num_LSTM = trial.suggest_int("num_LSTM", 1, 3), trial= trial).to(device)
    
    net = torch.compile(net)

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
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

        for in_volt, in_tensors, out in trainData:
            optimizer.zero_grad()
            output, _ = net(sequence_input=in_volt.to(device), scalar_input=in_tensors.to(device))
            loss = criterion(output, out.to(device))
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
    
    study_name = "LSTM_FNN_Studying"  # Unique identifier of the study.
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
