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
import optuna
import torch._dynamo as dynamo

#DEFINE CONSTANTS
# Reproducibility

random.seed(72323)
np.random.seed(72323)
torch.manual_seed(72323)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose=True

# Hyperparameters
# NUM_EPOCH = 165 #200
# BATCH_SIZE = 1024
# DECAY_EPOCH = 20
# DECAY_RATIO = 0.8
# LR_INI = 0.0026 #0.0026

# Set # of epochs to discard due to warmup 
DISCARD = 10

#Materials

materials = {
    "N87" : ["/scratch/gpfs/sw0123/N87/N87_R34.0X20.5X12.5_Data5_Seq2Scalar_Downsampled_FullFixed.json", 0],
    "3C90" : ["/scratch/gpfs/sw0123/3C90/3C90_TX-25-15-10_Data1_Seq2Scalar_Downsampled_FullFixed.json", 1],
    "3C94" : ["/scratch/gpfs/sw0123/3C94/3C94_TX-20-10-7_Data1_Seq2Scalar_Downsampled_FullFixed.json", 2]
}

# Select GPU as default device
device = torch.device("cuda")

# Define material head 

# optimize material head dimensions
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
class MagUniformer(nn.Module):
    def __init__(self, 
        input_size :int,
        max_seq_len :int,
        dim_val :int,  
        n_encoder_layers :int,
        n_heads :int,
        dropout_encoder,
        dropout_pos_enc,
        dim_feedforward_encoder :int,
        projected_variables :int,
        output_features :int,
        num_materials: int,
        trial,
        encoder_l: int): 

        #   Args:
        #    input_size: int, number of input variables. 1 if univariate.
        #    max_seq_len: int, length of the longest sequence the model will receive. Used in positional encoding. 
        #    dim_val: int, aka d_model. All sub-layers in the model produce outputs of dimension dim_val
        #    n_encoder_layers: int, number of stacked encoder layers in the encoder
        #    n_heads: int, the number of attention heads (aka parallel attention layers)
        #    dropout_encoder: float, the dropout rate of the encoder
        #    dropout_pos_enc: float, the dropout rate of the positional encoder
        #    dim_feedforward_encoder: int, number of neurons in the linear layer of the encoder
        #    projected_variables: int, number of extra scalar variables to project on to feature extractor
        #    output_features: int, number of features output by the feature extractor

        super().__init__() 

        self.n_heads = n_heads
        self.dim_val = dim_val
        self.num_materials = num_materials

        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, encoder_l),
            nn.GELU(),
            nn.Linear(encoder_l, dim_val))

        self.sos_token = nn.Parameter(torch.zeros(1, 1, dim_val))

        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc, max_len=max_seq_len)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="gelu",
            batch_first=True
            )
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_encoder_layers, norm=None)

        #Optimize projection layers
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

        self.heads = nn.Sequential(*[MaterialHead(output_features, trial, id) for id in range(num_materials)])

    def forward(self, in_seq: Tensor, vars: Tensor, materials: Tensor, device) -> Tensor:
        batch = in_seq.shape[0]
        x = self.encoder_input_layer(in_seq)

        #expand class token across batch direction
        sos_tokens = self.sos_token.expand(batch, -1, -1)
        x = torch.cat((sos_tokens, x), dim=1)

        x = self.positional_encoding_layer(x)
        x = self.encoder(x)

        # get 0th element in encoder's output sequence. 
        sos_out = x[:,0] # Bxseq_len x dim_val -> B x dim_val

        # B x 3 + B x dim_val --> B x (3+dim_val) --> B x output_features
        projector_out = self.projector(torch.cat((vars, sos_out),dim=1))
        
        output = torch.zeros(batch, 1, device=device)
        _, _, output = self.heads((projector_out, materials, output))

        return output

class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

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
    #print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(data) * 1e5:.5f}")

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
    error95 = np.percentile(Error_re, 95)
    print("95th percent: ", error95)
    print("99th percent: ", np.percentile(Error_re, 99))

    return error95

# Config the model training

def main(trial):
    print("BERT-Main loop entered!")

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

    heads = trial.suggest_int("heads", 4, 8, 4)

    # Setup network
    net = MagUniformer(
      dim_val=trial.suggest_int("dim_val", 16, 32, heads),
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=trial.suggest_int("encoder_layers", 1, 3),
      n_heads=heads,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=trial.suggest_int("dim_FFL", 8, 48), #old 32
      projected_variables=3,
      output_features=trial.suggest_int("out_f", 4, 16), #old 12
      num_materials=3,
      trial=trial,
      encoder_l=trial.suggest_int("encoder_l", 4, 40, 4)).to(device)
    
    net = torch.compile(net)
    # modes: default, reduce-overhead, max-autotune

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=MIN_LR) #0.0005, 160
    # PAST OPTIMS DONE WITH 165, 0.0053

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

        if (epoch_i+1)%35 == 0:
            intermediate_value = evaluate(net, validData, device)
            trial.report(intermediate_value, epoch_i)
            # Handle pruning based on the intermediate value.

            if trial.should_prune():
                raise optuna.TrialPruned()
        
        scheduler.step()

    return evaluate(net, validData, device)

if __name__ == "__main__":

    study_name = "FixedTransformerEd"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    # study = optuna.create_study(direction='minimize', 
    #                             pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=90),
    #                             study_name=study_name,
    #                             storage=storage_name,
    #                             load_if_exists=True)
    
    # study.optimize(main, n_trials=200, gc_after_trial=True)

    # trial = study.best_trial
    # print(trial)

    # joblib.dump(study, "study.pkl")

    
    # study_name = "Median95-Studying"  # Unique identifier of the study.
    # storage_name = "sqlite:///{}.db".format(study_name)

    # # study = optuna.create_study(direction='minimize', 
    # #                             pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=90),
    # #                             study_name=study_name,
    # #                             storage=storage_name)
    
    # # study.optimize(main, n_trials=100, gc_after_trial=True)

    # # trial = study.best_trial
    # # print(trial)
    # # joblib.dump(study, "study.pkl")

    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    #fig = optuna.visualization.plot_intermediate_values(study)
    #fig = optuna.visualization.plot_param_importances(study)
    fig = optuna.visualization.plot_optimization_history(study)

    fig.show()


