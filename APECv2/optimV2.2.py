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
import optuna

#exp.
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

#DEFINE CONSTANTS
# Reproducibility

random.seed(374327224)
np.random.seed(374327224)
torch.manual_seed(374327224) #OLD: 92323
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True) 

torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose=True

# Hyperparameters
# NUM_EPOCH = 638 #200
# BATCH_SIZE = 384
# LR_INI = 0.001374771109331439 #0.0026
# MIN_LR = 0.0004017121475967859

MUNOT = 1.25663706212e-6

# Set # of epochs to discard due to warmup 
DISCARD = 10

#Materials

#Training Materials 
#materials: material name : [filepath,ID, Bsat, freqPeak, mu_i]
materials = {
    "N87" : ["/scratch/gpfs/ed5754/N87/N87_R34.0X20.5X12.5_Data5_Seq2Scalar_Downsampled_FullFixed.json", 0, 490, 900,2200],
    "3C90" : ["/scratch/gpfs/ed5754/3C90/3C90_TX-25-15-10_Data1_Seq2Scalar_Downsampled_FullFixed.json", 1, 470, 600,2300],
    "3C94" : ["/scratch/gpfs/ed5754/3C94/3C94_TX-20-10-7_Data1_Seq2Scalar_Downsampled_FullFixed.json", 2, 470, 600,2300],
    "3F4" : ["/scratch/gpfs/ed5754/3F4/3F4_E-32-6-20-R_Data1_Seq2Scalar_Downsampled_FullFixed.json", 3, 410, 2000,900],
    "N30" : ["/scratch/gpfs/ed5754/N30/N30_22.1X13.7X6.35_Data1_Seq2Scalar_Downsampled_FullFixed.json", 4, 380, 200,4300],
    "N49" : ["/scratch/gpfs/ed5754/N49/N49_R16.0X9.6X6.3_Data1_Seq2Scalar_Downsampled_FullFixed.json", 5, 490, 1800,1500],
}

# Select GPU as default device
device = torch.device("cuda")

# Define material head - this is the framework for the material specific mapping networks
# This class takes in the number of features output from the shared feature extractor and
# also takes in the Material_ID referred to as task

# optimize material head dimensions
#this is safe, calling same trial suggest multiple times returns the same values!
class MaterialHead(nn.Module):
    def __init__(self, output_features, trial, task):
        super().__init__()
        self.task=task

        layers = trial.suggest_int("head_layers", 1, 4)
        layer = []
        in_f = output_features

        for i in range(layers):
                out_features = trial.suggest_int("head_units_{}".format(i), 2, 22)
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

# See https://discuss.pytorch.org/t/different-model-branches-for-different-samples-in-batch/151978/8 for
# guidance on how multi-material selection structure is achieved within batches

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
        parameter_variables :int,
        output_features :int,
        num_materials: int,
        encoder_l: int,
        trial
        ): 

        #   Args:
        #    input_size: int, number of input variables. 1 if univariate.
        #    max_seq_len: int, length of the longest sequence the model will receive. Used in positional encoding. 
        #    dim_val: int, aka d_model. The dimension of the transformer encoder
        #    n_encoder_layers: int, number of stacked encoder layers in the encoder
        #    n_heads: int, the number of attention heads (aka parallel attention layers)
        #    dropout_encoder: float, the dropout rate of the encoder
        #    dropout_pos_enc: float, the dropout rate of the positional encoder
        #    dim_feedforward_encoder: int, number of neurons in the linear layer of the encoder
        #    projected_variables: int, number of extra scalar variables to project on to feature extractor
        #    parameter_variables: int, number of extra scalar variables to pass to material networks
        #    output_features: int, number of features output by the feature extractor. This is the bottleneck section
        #    num_materials: int, number of materials specific mapping networks
        #    encoder_l: int, number of neurons in the input embedding layer, the intermediate dimension 

        super().__init__() 

        self.n_heads = n_heads
        self.dim_val = dim_val
        self.num_materials = num_materials

        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, encoder_l),
            nn.GELU(),
            nn.Linear(encoder_l, dim_val))

        # Create Start of Sequence token, similar to BERT/ViT approaches to distill sequence
        # This is learnable -- does not pass through input embedding network but has same dimension for encoder
        self.sos_token = nn.Parameter(torch.zeros(1, 1, dim_val))

        # Basic Positional Encoding and Transformer Encoder setup.
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

        # Setup projection network to incorporate the extra scalar variables
        # Input dimension is dim_val of model + the number of extra scalar inputs
        # Output dimension is the bottleneck size before the material speciifc mapping networks

        proj_layers = trial.suggest_int("proj_layers", 1, 4)
        proj = []
        proj_in_f = dim_val + projected_variables

        for i in range(proj_layers):
                out_features = trial.suggest_int("proj_units_{}".format(i), 6, 38)
                proj.append(nn.Linear(proj_in_f, out_features))
                proj.append(nn.GELU())
                proj_in_f = out_features

        proj.append(nn.Linear(proj_in_f, output_features))
        self.projector = nn.Sequential(*proj)
        
        # Unique implementation: create sequential list of the material mapping networks
        # Thus, data gets passed through from one material network to the next, and each
        # network selectively activates over the data with the matching ID
        # Effecient, torch.compile() compatible implementation 
        input_features = output_features + parameter_variables
        self.heads = nn.Sequential(*[MaterialHead(input_features, trial, id) for id in range(num_materials)])

    def forward(self, in_seq: Tensor, proj_vars: Tensor, param_vars: Tensor, materials: Tensor, device) -> Tensor:
        batch = in_seq.shape[0]
        x = self.encoder_input_layer(in_seq)

        #expand class token across batch direction. Then, append to expanded input sequence
        sos_tokens = self.sos_token.expand(batch, -1, -1)
        x = torch.cat((sos_tokens, x), dim=1)
        #x: batch x 129 x dim_val

        # now, add PE's and pass through encoder
        x = self.positional_encoding_layer(x)
        x = self.encoder(x)

        # get 0th element in encoder's output sequence. 
        sos_out = x[:,0] # B x seq_len x dim_val -> B x dim_val

        # Pass thru projection network
        # B x 1 + B x dim_val --> B x (1+dim_val) --> B x output_features
        projector_out = self.projector(torch.cat((proj_vars, sos_out),dim=1))

        # Now, add the parameter variables to pass through the material networks
        # B x 2 + B x out_features --> B x (paramVars+out_features)
        mat_specific_in = torch.cat((param_vars, projector_out), dim=1)

        # create blank output tensor to fill with data 
        output = torch.zeros(batch, 1, device=device)

        #pass all the info through the material networks, only maintain the output
        _, _, output = self.heads((mat_specific_in, materials, output))

        # B x 1 (core loss)
        return output

# Standard Positional Encoder implementation
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

#Count learnable parameters in entire model 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load the constituent datasets, combine and output
def get_datasets(materials):
    datasets = []

    for M, info in materials.items():
        datasets.append(load_dataset(info))

    return torch.utils.data.ConcatDataset(datasets)

# Load individual dataset, taking in ID and path
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
    Power = Power.reshape((-1,1))

    # Combine the T/Hdc scalars into one tensor with two columns (LEN x 2)
    combinedInput = np.concatenate((Temp,Hdc),axis=1)
    parameterTensor = torch.from_numpy(combinedInput).float().view(-1, 2)

    # hold frequency separately to be passed thru projection network
    freqTensor = torch.from_numpy(Freq).float().view(-1,1)

    #Reshape Bdc to N*1*1 then to N*128*1, with Hdc values repeated along the 128 dimension
    BdcTensor = torch.from_numpy(Bdc).float().view(-1,1,1).expand(-1, data_length, -1)

    in_B = torch.from_numpy(B).float().view(-1, data_length, 1)

    #Add BdcTensor to the in_B: for each sequence, a single Bdc value will be added to all points
    in_B_adjusted = torch.add(in_B, BdcTensor)
    in_B_adjusted = torch.div(in_B_adjusted, bSaturation)

    out = torch.from_numpy(Power).float().view(-1,1)

    # put a label with each datapoint, corresponding to the material ID
    labels = torch.full((out.size()), label)

    return torch.utils.data.TensorDataset(in_B_adjusted, freqTensor, parameterTensor, labels, out)

# Function taking in network,data,device for evaluation 
def evaluate(net, data, device, sprint):
    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for in_B, freqTensor, parameterTensor, labels, out in data:
            y_pred.append(net(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
            param_vars=parameterTensor.to(device), materials=labels.to(device), device=device))

            y_meas.append(out.to(device))

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(data) * 1e5:.5f}")

    yy_pred = 10**(y_pred.cpu().numpy())
    yy_meas = 10**(y_meas.cpu().numpy())
    
    # Relative Error
    Error_re = abs(yy_pred-yy_meas)/abs(yy_meas)*100

    if sprint==1:
        Error_re_avg = np.mean(Error_re)
        #Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
        Error_re_max = np.max(Error_re)
        print(f"Relative Error: {Error_re_avg:.8f}")
        #print(f"RMS Error: {Error_re_rms:.8f}")
        print(f"MAX Error: {Error_re_max:.8f}")
        print("95th percent: ", np.percentile(Error_re, 95))
        print("99th percent: ", np.percentile(Error_re, 99))

    return np.percentile(Error_re, 95)

# Config the model training
def objective(trial):
    print("Main loop entered!")
    BATCH_SIZE = trial.suggest_int("batch", 256, 768, 128)

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

    NUM_EPOCH = trial.suggest_int("epoch", 220, 650)
    LR_INI = trial.suggest_float("LR", 0.0009, 0.006)
    MIN_LR = trial.suggest_float("MIN_LR", 0.00005, 0.0008)

    heads = trial.suggest_int("heads", 4, 8, 4)

    # Setup network
    net = MagUniformer(
      dim_val=trial.suggest_int("dim_val", 8, 32, heads),
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=trial.suggest_int("encoder_layers", 1, 3),
      n_heads=heads,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=trial.suggest_int("dim_FFL", 12, 36),
      projected_variables=1,
      parameter_variables=2,
      output_features=2,
      num_materials=6,
      encoder_l=trial.suggest_int("encoder_inp", 8, 36),
      trial=trial).to(device)
    
    #not optimizing encoder_l.
    
    net = torch.compile(net)

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 

    # Use this COS annealing LR for better performance!
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=MIN_LR) #0.0005, 160
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1.1, eta_min=0.0006)

    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        #optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))


        for in_B, freqTensor, parameterTensor, labels, out in trainData:
            optimizer.zero_grad()
            output = net(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
                         param_vars=parameterTensor.to(device), materials=labels.to(device), device=device)
            
            loss = criterion(output, out.to(device))
            loss.backward()
 
            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.25)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        #evaluate(net, validData, device, 1)

        if (epoch_i+1)%30 == 0:
            print("On epoch ", epoch_i)
            intermediate_value = evaluate(net, validData, device, 1)
            trial.report(intermediate_value, epoch_i)
            # Handle pruning based on the intermediate value.

            if trial.should_prune():
                raise optuna.TrialPruned()
        
        #Collect best results from last 30 epochs!
        if (epoch_i == NUM_EPOCH-30):
            bestYet = evaluate(net, validData, device, 0)
            bestEpoch = epoch_i
        if (epoch_i > NUM_EPOCH-30):
            #supress print statements 
            intermediate_value = evaluate(net, validData, device, 0)

            if (intermediate_value < bestYet):
                bestYet = intermediate_value
                bestEpoch = epoch_i
        
        scheduler.step()

    print("Best Epoch: ", bestEpoch, "with a value of ", bestYet)
    return bestYet

if __name__ == "__main__":

    study_name = "SizeTwoStrictly"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction='minimize', 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=70),
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    
    study.optimize(objective, n_trials=150, gc_after_trial=True)
