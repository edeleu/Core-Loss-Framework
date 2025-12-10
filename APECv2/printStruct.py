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

TRANSFER_LEARN = False
TRANSFER_LR = 0.001
TRANSFER_MINLR = 0.0001
TRANSFER_EPOCH = 100
TRANSFER_DATA = 120
TRANSFER_BATCH = 10

# Set # of epochs to discard due to warmup 
DISCARD = 10

#Materials

#Training Materials 
materials = {
    "N87" : ["/scratch/gpfs/ed5754/N87/N87_R34.0X20.5X12.5_Data5_Seq2Scalar_Downsampled_FullFixed.json", 0],
    "3C90" : ["/scratch/gpfs/ed5754/3C90/3C90_TX-25-15-10_Data1_Seq2Scalar_Downsampled_FullFixed.json", 1],
    "3C94" : ["/scratch/gpfs/ed5754/3C94/3C94_TX-20-10-7_Data1_Seq2Scalar_Downsampled_FullFixed.json", 2]
}

transferMaterial = {
    "N87" : ["/scratch/gpfs/ed5754/", 11]
}

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
            nn.Linear(input_size, 15),
            nn.GELU(),
            nn.Linear(15, 18),
            nn.GELU(),
            nn.Linear(18, 1)
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
            nn.Linear(dim_val + projected_variables, 34),
            nn.GELU(),
            nn.Linear(34, 11),
            nn.GELU(),
            nn.Linear(11,36),
            nn.GELU(),
            nn.Linear(36, 22),
            nn.GELU(),
            nn.Linear(22, output_features)
        )
        
        # Unique implementation: create sequential list of the material mapping networks
        # Thus, data gets passed through from one material network to the next, and each
        # network selectively activates over the data with the matching ID
        # Effecient, torch.compile() compatible implementation 
        input_features = output_features + parameter_variables
        self.outFeatures = input_features

        self.heads = nn.Sequential(*[MaterialHead(input_features, id) for id in range(num_materials)])

        # EXPERIMENTAL: T/Hdc processing network 
        self.input_param_layer = nn.Sequential(
            nn.Linear(parameter_variables, 3),
            nn.GELU(),
            nn.Linear(3, parameter_variables),
        )

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

        # EXPERIMENTAL: Param processing network 
        proc_params = self.input_param_layer(param_vars)

        # Now, add the parameter variables to pass through the material networks
        # B x 2 + B x out_features --> B x (paramVars+out_features)
        mat_specific_in = torch.cat((proc_params, projector_out), dim=1)

        if (~TRANSFER_LEARN):
            # create blank output tensor to fill with data 
            output = torch.zeros(batch, 1, device=device)

            #pass all the info through the material networks, only maintain the output
            _, _, output = self.heads((mat_specific_in, materials, output))

            # B x 1 (core loss)
            return output
        else:
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

# Load individual dataset, taking in ID and path
def load_dataset(info, data_length=128):
    path = info[0]
    label = info[1]

    # Load .json Files
    with open(path,'r') as load_f:
        DATA = json.load(load_f)

    B = DATA['B_Field']
    B = np.array(B)
    Freq = DATA['Frequency']
    Freq = np.log10(Freq)
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

    # Combine the T/Hdc scalars into one tensor with two columns (LEN x 2)
    combinedInput = np.concatenate((Temp,Hdc),axis=1)
    parameterTensor = torch.from_numpy(combinedInput).float().view(-1, 2)

    # hold frequency separately to be passed thru projection network
    freqTensor = torch.from_numpy(Freq).float().view(-1,1)

    in_B = torch.from_numpy(B).float().view(-1, data_length, 1)

    out = torch.from_numpy(Power).float().view(-1,1)

    # put a label with each datapoint, corresponding to the material ID
    labels = torch.full((out.size()), label)

    return torch.utils.data.TensorDataset(in_B, freqTensor, parameterTensor, labels, out)

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

    return np.percentile(Error_re, 95), Error_re_avg
# Config the model training
def main():
    print("Main loop entered!")

    # Record initial time
    start_time = time.time()

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

    # Setup network
    net = MagUniformer(
      dim_val=24,
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=3,
      n_heads=8,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=33,
      projected_variables=1,
      parameter_variables=2,
      output_features=14,
      num_materials=3,
      encoder_l=32).to(device)
    
    net = torch.compile(net)

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 

    # Use this COS annealing LR for better performance!
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=MIN_LR) #0.0005, 160
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1.1, eta_min=0.0006)
    
    # Create list to store epoch times
    times=[]

    Errors = []

    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        #optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))

        torch.cuda.synchronize()
        start_epoch = time.time()

        for in_B, freqTensor, parameterTensor, labels, out in trainData:
            optimizer.zero_grad()
            output = net(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
                         param_vars=parameterTensor.to(device), materials=labels.to(device), device=device)
            
            loss = criterion(output, out.to(device))
            loss.backward()
 
            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.25)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Record epoch time 
        torch.cuda.synchronize()
        end_epoch = time.time()
        times.append(end_epoch-start_epoch)

        # for first N-100 epochs, continue as usual
        if (epoch_i < NUM_EPOCH-100):
            if (epoch_i+1)%2 == 0:
                with torch.no_grad():
                    net.eval()
                    epoch_valid_loss = 0
                    for in_B, freqTensor, parameterTensor, labels, out in valid_loader:
                        output = net(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
                         param_vars=parameterTensor.to(device), materials=labels.to(device), device=device)

                        loss = criterion(output, out.to(device))
                        epoch_valid_loss += loss.item()

                print(f"Epoch {epoch_i+1:2d} "
                f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
                f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")
                print()
                evaluate(net, valid_loader, device)
        # Afterwards, save model at every epoch and record validation error as an array. 
        else:
            print(epoch_i, " now")

            with torch.no_grad():
                    net.eval()
                    epoch_valid_loss = 0
                    for in_B, freqTensor, parameterTensor, labels, out in valid_loader:
                        output = net(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
                         param_vars=parameterTensor.to(device), materials=labels.to(device), device=device)
                        
                        loss = criterion(output, out.to(device))
                        epoch_valid_loss += loss.item()

            print(f"Epoch {epoch_i+1:2d} "
            f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
            f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")
            print()

            torch.save(net.state_dict(), "/scratch/gpfs/ed5754/SmallModel/UnifiedTransformer" + str(epoch_i) + ".sd")
            Errors.append(evaluate(net, valid_loader, device))

        # LR Step
        scheduler.step()

    elapsed = time.time() - start_time
    print(f"Total Time Elapsed: {elapsed}")    
    print(f"Average time per Epoch: {sum(times[DISCARD:])/(NUM_EPOCH-DISCARD)}")
    
    # Save the model parameters
    #torch.save(net.state_dict(), "/scratch/gpfs/ed5754/UnifiedMulti.sd")
    print("Training finished! Model is saved!")

    # write error values to spreadsheet, can take a while. 
    with open("ErrorsSmall.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(Errors)

# Similar to training, but loading existing model save.
def inference():
    device = torch.device("cuda")

    # Load dataset
    dataset = get_datasets(materials)

    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, **kwargs)
    testData = list(train_loader)
    
    ##NETWORK SETUP
        
    net = MagUniformer(
      dim_val=24,
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=3,
      n_heads=8,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=33,
      projected_variables=1,
      parameter_variables=2,
      output_features=14,
      num_materials=3,
      encoder_l=32).to(device)
    
    net = torch.compile(net)

    state_dict = torch.load('/scratch/gpfs/ed5754/UnifiedTransformer623.sd')
    net.load_state_dict(state_dict, strict=True)
    
    net.eval()
    print("Model is loaded!")
    
    evaluate(net, testData, device)

# Create Transfer Learning Code:

# Only supports transfer learned materials (all at once)
# Load from folder the model 

def transferLearn():
    # Load existing trained model

    unifiedNet = MagUniformer(
      dim_val=24,
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=3,
      n_heads=8,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=33,
      projected_variables=1,
      parameter_variables=2,
      output_features=14,
      num_materials=3,
      encoder_l=32).to(device)
    
    unifiedNet = torch.compile(unifiedNet)

    state_dict = torch.load('/scratch/gpfs/ed5754/UnifiedTransformer623.sd')
    unifiedNet.load_state_dict(state_dict, strict=True)

    # Lock all of the parameters just in case!
    for param in unifiedNet.parameters():
        param.requires_grad = False
    
    unifiedNet.eval()

    # Create new transferring network 
    numFeatures = 2 + 14

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

                output = transferNetwork(unifiedNet(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
                            param_vars=parameterTensor.to(device), materials=labels.to(device), device=device))
                
                loss = criterion(output, out.to(device))
                loss.backward()
    
                #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.25)
                optimizer.step()
                epoch_train_loss += loss.item()

            with torch.no_grad():
                transferNetwork.eval()
                epoch_valid_loss = 0
                for in_B, freqTensor, parameterTensor, labels, out in valid_loader:
                    output = transferNetwork(unifiedNet(in_seq=in_B.to(device), proj_vars=freqTensor.to(device), 
                        param_vars=parameterTensor.to(device), materials=labels.to(device), device=device))

                    loss = criterion(output, out.to(device))
                    epoch_valid_loss += loss.item()

                print(f"Epoch {epoch_i+1:2d} "
                f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
                f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")
                print()
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
        
        results.append([DATA_SIZE, o1Best, o2Best])

    with open("TransferResults.csv", "w") as f:
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
    if (x==1): return MaterialHead(input_size=numFeatures, task=11)

    network = MaterialHead(input_size=numFeatures, task=11)
    network = torch.compile(network)
    state_dict = torch.load('/scratch/gpfs/ed5754/UnifiedTransformer623.sd')
    network.load_state_dict(state_dict, strict=True)
    
    return network

net = MagUniformer(
    dim_val=24,
    input_size=1, 
    max_seq_len=129,
    n_encoder_layers=3,
    n_heads=8,
    dropout_encoder=0.0, 
    dropout_pos_enc=0.0,
    dim_feedforward_encoder=33,
    projected_variables=1,
    parameter_variables=2,
    output_features=14,
    num_materials=3,
    encoder_l=32).to(device)

print(net)
print("okay next one")
print()
net = torch.compile(net)
print(net)
state_dict = torch.load('/scratch/gpfs/ed5754/GeneralFramework466.sd')
print(state_dict)

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
    
myHead = MyHead(14, 11)
print(myHead.state_dict())


# head.0.weight
# head.0.bias

# head.2.weight
# head.2.bias

# head.4.weight
# head.4.bias

#_orig_mod.heads.5.(head.etc)

# trained = torch.load('model.pth')
# tryrained_trim = {k:v for k, v in trained.items() if not k.startswith('fc')}

pretrained_dict = torch.load('/scratch/gpfs/ed5754/GeneralFramework466.sd') #pretrained model keys
processed_dict = {}

for k in pretrained_dict.keys():
    if("_orig_mod.heads.0" in k):
        decomposed = k.split(".")
        newKey = ".".join(decomposed[3:])
        print(newKey)
        processed_dict[newKey] = pretrained_dict[k]

myHead = MyHead(14, 11)
myHead.load_state_dict(processed_dict, strict=False)
torch.save(myHead.state_dict(), "/scratch/gpfs/ed5754/ExperimentalHead.sd")
