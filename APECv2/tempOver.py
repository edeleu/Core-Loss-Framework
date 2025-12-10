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

#DEFINE CONSTANTS
# Reproducibility

random.seed(374327224)
np.random.seed(374327224)
torch.manual_seed(374327224) #OLD: 92323
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

TRANSFER_LEARN = True
TRANSFER_LR = 0.0023133072515642535
TRANSFER_MINLR = 0.0005331871134697933
TRANSFER_EPOCH = 240
TRANSFER_BATCH = 160

# Set # of epochs to discard due to warmup 
DISCARD = 10

#Materials

#Training Materials 
materials = {
    "N87" : ["/scratch/gpfs/ed5754/N87/N87_R34.0X20.5X12.5_Data5_Seq2Scalar_Downsampled_FullFixed.json", 0, 490, 900,2200]
}

#materials: material name : [filepath,ID, Bsat, freqPeak, mu_i]
# transferMaterial = {
#     "3E6" : ["/scratch/gpfs/ed5754/3E6_TL/3E6_TX-22-14-6.4_Data1_Seq2Scalar_Downsampled_Full.json", 11, 390,50,12000]
# }

transferMaterial = {
    "78" : ["/scratch/gpfs/ed5754/78_TL/78_0076_Data1_Seq2Scalar_Downsampled_Full.json", 11, 480,900,2300]
}

#3E6_TX-22-14-6.4_Data1_Seq2Scalar_Downsampled_FullFixed.json

# #Testing Materials
# materials = {
#     "N87" : ["/scratch/gpfs/ed5754/N87/N87_R34.0X20.5X12.5_Data5_Seq2Scalar_Downsampled_FullTest.json", 0]
# }

# Select GPU as default device
device = torch.device("cuda")

# Define material head - this is the framework for the material specific mapping networks
# This class takes in the number of features output from the shared feature extractor and
# also takes in the Material_ID referred to as task
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
        encoder_l: int
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
        self.projector =  nn.Sequential(
            nn.Linear(dim_val + projected_variables, 28),
            nn.GELU(),
            nn.Linear(28, 33),
            nn.GELU(),
            nn.Linear(33,32),
            nn.GELU(),
            nn.Linear(32, 27),
            nn.GELU(),
            nn.Linear(27, output_features)
        )
        
        # Unique implementation: create sequential list of the material mapping networks
        # Thus, data gets passed through from one material network to the next, and each
        # network selectively activates over the data with the matching ID
        # Effecient, torch.compile() compatible implementation 
        input_features = output_features + parameter_variables
        self.heads = nn.Sequential(*[MaterialHead(input_features, id) for id in range(num_materials)])

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

        return mat_specific_in

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
def evaluate(net, data, device):
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
    #print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(data) * 1e5:.5f}")

    yy_pred = 10**(y_pred.cpu().numpy())
    yy_meas = 10**(y_meas.cpu().numpy())
    
    # Relative Error
    Error_re = abs(yy_pred-yy_meas)/abs(yy_meas)*100
    Error_re_avg = np.mean(Error_re)
    Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    Error_re_max = np.max(Error_re)
    #print(f"Relative Error: {Error_re_avg:.8f}")
    #print(f"RMS Error: {Error_re_rms:.8f}")
    #print(f"MAX Error: {Error_re_max:.8f}")
    #print("95th percent: ", np.percentile(Error_re, 95))
    #print("99th percent: ", np.percentile(Error_re, 99))

    return Error_re

def evaluateTransfer(OGNET, net, data, device):
    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for in_B, freqTensor, parameterTensor, labels, out in data:
            y_pred.append(net(OGNET(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
            param_vars=parameterTensor.to(device), materials=labels.to(device), device=device)))

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
    # print(f"Relative Error: {Error_re_avg:.8f}")
    # print(f"RMS Error: {Error_re_rms:.8f}")
    # print(f"MAX Error: {Error_re_max:.8f}")
    # print("95th percent: ", np.percentile(Error_re, 95))
    # print("99th percent: ", np.percentile(Error_re, 99))

    return np.percentile(Error_re, 95), Error_re_avg

def evaluateTransferFull(OGNET, net, data, device):
    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for in_B, freqTensor, parameterTensor, labels, out in data:
            y_pred.append(net(OGNET(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
            param_vars=parameterTensor.to(device), materials=labels.to(device), device=device)))

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
    # print(f"Relative Error: {Error_re_avg:.8f}")
    # print(f"RMS Error: {Error_re_rms:.8f}")
    # print(f"MAX Error: {Error_re_max:.8f}")
    # print("95th percent: ", np.percentile(Error_re, 95))
    # print("99th percent: ", np.percentile(Error_re, 99))

    return np.percentile(Error_re, 95), Error_re_avg, Error_re_max, np.percentile(Error_re, 99)
# Create Transfer Learning Code:

# Only supports transfer learned materials (all at once)
# Load from folder the model 

def transferLearn():
    # Load existing trained model

    unifiedNet = MagUniformer(
      dim_val=20,
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=2,
      n_heads=4,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=40,
      projected_variables=1,
      parameter_variables=2,
      output_features=11,
      num_materials=6,
      encoder_l=32).to(device)
    
    unifiedNet = torch.compile(unifiedNet)

    state_dict = torch.load('/scratch/gpfs/ed5754/GeneralFramework466.sd')
    unifiedNet.load_state_dict(state_dict, strict=True)

    # Lock all of the parameters just in case!
    for param in unifiedNet.parameters():
        param.requires_grad = False
    
    unifiedNet.eval()

    # Create new transferring network 
    numFeatures = 2 + 11

    # Load dataset
    dataset = get_datasets(transferMaterial)
    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}

    results=[]
    # start loop over varying amounts of data
    for DATA_SIZE in range(100,5000,100):
        
        # Split the dataset
        train_size = DATA_SIZE
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRANSFER_BATCH, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=TRANSFER_BATCH, shuffle=False, **kwargs)
        trainData = list(train_loader)

        transferNetwork = loadMaterialNetwork(1, numFeatures)
        Tstate_dict = torch.load('/scratch/gpfs/ed5754/ExperimentalHead0.sd')
        transferNetwork.load_state_dict(Tstate_dict, strict=False)
#        transferNetwork = loadMaterialNetwork(1, numFeatures)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(transferNetwork.parameters(), lr=TRANSFER_LR) 

        # Use this COS annealing LR for better performance!
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRANSFER_EPOCH, eta_min=TRANSFER_MINLR) #0.0005, 160

        for epoch_i in range(TRANSFER_EPOCH):

            # Train for one epoch
            epoch_train_loss = 0
            transferNetwork.train()

            for in_B, freqTensor, parameterTensor, labels, out in trainData:
                optimizer.zero_grad()

                unifiedOut = unifiedNet(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
                            param_vars=parameterTensor.to(device), materials=labels.to(device), device=device)
                
                output = transferNetwork(unifiedOut)
                            
                loss = criterion(output, out.to(device))
                loss.backward()

                #torch.nn.utils.clip_grad_norm_(transferNetwork.parameters(), max_norm=1.25)
                optimizer.step()
                epoch_train_loss += loss.item()

            with torch.no_grad():
                transferNetwork.eval()
                epoch_valid_loss = 0
                if (epoch_i % 10 ==0):
                    for in_B, freqTensor, parameterTensor, labels, out in valid_loader:
                        output = transferNetwork(unifiedNet(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
                            param_vars=parameterTensor.to(device), materials=labels.to(device), device=device))

                        loss = criterion(output, out.to(device))
                        epoch_valid_loss += loss.item()

                    print(f"Epoch {epoch_i+1:2d} "
                    f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
                    f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")
                    print()
                o1, o2, o3, o4 = evaluateTransferFull(unifiedNet, transferNetwork, valid_loader, device)

            # LR Step
            scheduler.step()

            if (epoch_i == 20):
                o1Best = o1
                o2Best = o2
                o3Best = o3
                o4Best = o4
            elif (epoch_i > TRANSFER_EPOCH-218):
                if (o1<o1Best):
                    o1Best = o1
                    o2Best = o2
                    o3Best = o3
                    o4Best = o4
        
        print(o1Best)
        print(o2Best)
        print(o3Best)
        results.append([DATA_SIZE, o1Best, o2Best, o3Best, o4Best])

    with open("Mat77_TLfromN87_ResultsEarlyStop.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)


# CURRENTLY: Instantiating a brand new material network
# TODO: AVG ALL EXISTING MATERIAL NETWORKS AND USE THIS AS A STARTING POINT!

#SHOW COMPARISON: REFERENCE VS TL. MODELS WITH SINGLE MATERIAL --> SAME NETWORK FOR SIMPLICITY
#FINETUNE: DataLimit, Epochs (fixed)
#3E6, N30
#Maintain Ratio of dataset, not only training performance but WHOLE DATASET performance!

# USE RANDOM SPLIT function to define numPoints
# Idea: for loop training, incrementing numPoints by 10/20 and retraining model, collecting best performance run on entire dataset:
# plot this graph. do the same thing for reference and transferlearned

#prev: just 50 datapoints to get decent results!
#Get best 95th from last 20 epochs of training (just in case)

#1 for blank transfer. 2 for loading transfer
def loadMaterialNetwork(x, numFeatures):
    if (x==1): return MyHead(input_size=numFeatures, task=11).to(device)

    network = MaterialHead(input_size=numFeatures, task=11)
    network = torch.compile(network)
    state_dict = torch.load('/scratch/gpfs/ed5754/UnifiedTransformer623.sd')
    network.load_state_dict(state_dict, strict=True)
    
    return network

#inspired by https://discuss.pytorch.org/t/average-each-weight-of-two-models/77008
def avgMatNetworks():
    taskID = 11 # Arbitrary choice for transfer learning, we don't use this feature. Potentially implement this later on
    numFeatures = 14 # Update this depending on exact model scenario 
    
    sd0 = torch.load('/scratch/gpfs/ed5754/ExperimentalHead0.sd')
    sd1 = torch.load('/scratch/gpfs/ed5754/ExperimentalHead1.sd')
    sd2 = torch.load('/scratch/gpfs/ed5754/ExperimentalHead2.sd')
    sd3 = torch.load('/scratch/gpfs/ed5754/ExperimentalHead3.sd')
    sd4 = torch.load('/scratch/gpfs/ed5754/ExperimentalHead4.sd')
    sd5 = torch.load('/scratch/gpfs/ed5754/ExperimentalHead5.sd')
    
    # Average all parameters
    for key in sd1:
        sd2[key] = (sd0[key] + sd1[key] + sd2[key] + sd3[key] + sd4[key] + sd5[key]) / 6.0

    # Save the averaged material model!
    torch.save(sd2, "/scratch/gpfs/ed5754/AVGNetwork.sd")

#Open Q: how to save MatNetworks, do they need to be compiled?
# Does taskID break with averaging? what happens here... 

def singleTL(trial):
    # Load existing trained model

    TLRINI = trial.suggest_float("LR", 0.002, 0.1)
    TLRMin = trial.suggest_float("MIN_LR", 0.0001, 0.001)

    TrEpoch = trial.suggest_int("epochs", 30, 300)
    TBatch = trial.suggest_int("batch", 8, 256, 8)

    unifiedNet = MagUniformer(
      dim_val=20,
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=2,
      n_heads=4,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=40,
      projected_variables=1,
      parameter_variables=2,
      output_features=11,
      num_materials=6,
      encoder_l=32).to(device)
    
    unifiedNet = torch.compile(unifiedNet)

    state_dict = torch.load('/scratch/gpfs/ed5754/GeneralFramework466.sd')
    unifiedNet.load_state_dict(state_dict, strict=True)

    # Lock all of the parameters just in case!
    for param in unifiedNet.parameters():
        param.requires_grad = False
    
    unifiedNet.eval()

    # Create new transferring network 
    numFeatures = 2 + 11

    dataset = get_datasets(transferMaterial)

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TBatch, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=TBatch, shuffle=False, **kwargs)

    trainData = list(train_loader)

    transferNetwork = loadMaterialNetwork(1, numFeatures)
    # Tstate_dict = torch.load('/scratch/gpfs/ed5754/ExperimentalHead1.sd')
    # transferNetwork.load_state_dict(Tstate_dict, strict=False)

    #print(count_parameters(transferNetwork))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(transferNetwork.parameters(), lr=TLRINI) 

    # Use this COS annealing LR for better performance!
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrEpoch, eta_min=TLRMin)

    for epoch_i in range(TrEpoch):

        # Train for one epoch
        epoch_train_loss = 0
        transferNetwork.train()

        for in_B, freqTensor, parameterTensor, labels, out in trainData:
            optimizer.zero_grad()

            unifiedOut = unifiedNet(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
                        param_vars=parameterTensor.to(device), materials=labels.to(device), device=device)
            
            output = transferNetwork(unifiedOut)
                        
            loss = criterion(output, out.to(device))
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(transferNetwork.parameters(), max_norm=1.25)
            optimizer.step()
            epoch_train_loss += loss.item()

        with torch.no_grad():
            transferNetwork.eval()
            #print(f"Epoch {epoch_i+1:2d} ")
            #print()
            o1, o2 = evaluateTransfer(unifiedNet, transferNetwork, valid_loader, device)

        # LR Step
        scheduler.step()

        if (epoch_i == 20):
            o1Best = o1
            o2Best = o2
        elif (epoch_i > TRANSFER_EPOCH-20):
            if (o1<o1Best):
                o1Best = o1
                o2Best = o2
    print(o2Best)
    return o1Best

def limitedTL(trial):
        # Load existing trained model

    TLRINI = trial.suggest_float("LR", 0.002, 0.1)
    TLRMin = trial.suggest_float("MIN_LR", 0.0001, 0.001)

    TrEpoch = trial.suggest_int("epochs", 30, 300)
    TBatch = trial.suggest_int("batch", 8, 256, 8)

    unifiedNet = MagUniformer(
      dim_val=20,
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=2,
      n_heads=4,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=40,
      projected_variables=1,
      parameter_variables=2,
      output_features=11,
      num_materials=6,
      encoder_l=32).to(device)
    
    unifiedNet = torch.compile(unifiedNet)
    state_dict = torch.load('/scratch/gpfs/ed5754/GeneralFramework466.sd')
    unifiedNet.load_state_dict(state_dict, strict=True)

    # Lock all of the parameters just in case!
    for param in unifiedNet.parameters():
        param.requires_grad = False
    
    unifiedNet.eval()

    # Create new transferring network 
    numFeatures = 2 + 11

    dataset = get_datasets(transferMaterial)

    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    # Split the dataset
    train_size = 1350
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRANSFER_BATCH, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=TRANSFER_BATCH, shuffle=False, **kwargs)
    trainData = list(train_loader)

#ExperimentalHead6.sd is AVG
    # listIndex = trial.suggest_int("pretrain", 0, 6)
    transferNetwork = loadMaterialNetwork(1, numFeatures)
    Tstate_dict = torch.load('/scratch/gpfs/ed5754/ExperimentalHead1.sd')
    transferNetwork.load_state_dict(Tstate_dict, strict=False)

    #print(count_parameters(transferNetwork))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(transferNetwork.parameters(), lr=TLRINI) 

    # Use this COS annealing LR for better performance!
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrEpoch, eta_min=TLRMin)

    for epoch_i in range(TrEpoch):

        # Train for one epoch
        epoch_train_loss = 0
        transferNetwork.train()

        for in_B, freqTensor, parameterTensor, labels, out in trainData:
            optimizer.zero_grad()

            unifiedOut = unifiedNet(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
                        param_vars=parameterTensor.to(device), materials=labels.to(device), device=device)
            
            output = transferNetwork(unifiedOut)
                        
            loss = criterion(output, out.to(device))
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(transferNetwork.parameters(), max_norm=1.25)
            optimizer.step()
            epoch_train_loss += loss.item()

        with torch.no_grad():
            transferNetwork.eval()
            epoch_valid_loss=0

            if (epoch_i % 10 ==0):
                    for in_B, freqTensor, parameterTensor, labels, out in valid_loader:
                        output = transferNetwork(unifiedNet(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
                            param_vars=parameterTensor.to(device), materials=labels.to(device), device=device))

                        loss = criterion(output, out.to(device))
                        epoch_valid_loss += loss.item()

                    print(f"Epoch {epoch_i+1:2d} "
                    f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
                    f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")
                    print()

            #print(f"Epoch {epoch_i+1:2d} ")
            #print()
            o1, o2 = evaluateTransfer(unifiedNet, transferNetwork, valid_loader, device)

        # LR Step
        scheduler.step()

        if (epoch_i == 20):
            o1Best = o1
            o2Best = o2
        elif (epoch_i > TRANSFER_EPOCH-20):
            if (o1<o1Best):
                o1Best = o1
                o2Best = o2
    print(o2Best)
    return o1Best

def optimizeMe():
    study_name = "1k78TL"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction='minimize',
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    
    study.optimize(singleTL, n_trials=150, gc_after_trial=True)

def existingInference(SD):
    device = torch.device("cuda")

    # Load dataset
    dataset = get_datasets(materials)

    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, **kwargs)
    testData = list(train_loader)
    
    ##NETWORK SETUP
        
    unifiedNet = MagUniformer(
      dim_val=20,
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=2,
      n_heads=4,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=40,
      projected_variables=1,
      parameter_variables=2,
      output_features=11,
      num_materials=6,
      encoder_l=32).to(device)
    
    unifiedNet = torch.compile(unifiedNet)

    state_dict = torch.load('/scratch/gpfs/ed5754/GeneralFramework466.sd')
    unifiedNet.load_state_dict(state_dict, strict=True)

    transferNetwork = loadMaterialNetwork(1, 13)
    Tstate_dict = SD
    transferNetwork.load_state_dict(Tstate_dict, strict=False)
    
    transferNetwork.eval()
    print("Model is loaded!")
    
    evaluateTransfer(unifiedNet, transferNetwork, testData, device)

if __name__ == "__main__":
    optimizeMe()
    # for i in range(0,6):
    #     existingInference(torch.load('/scratch/gpfs/ed5754/ExperimentalHead' + str(i) +'.sd'))
    #avgMatNetworks()
    #transferLearn()
