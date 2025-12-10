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

#DEFINE CONSTANTS
# Reproducibility

# random.seed(1)
# np.random.seed(1)
# torch.manual_seed(1)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose=True

# Hyperparameters
NUM_EPOCH = 165*4 #200
BATCH_SIZE = 1024
DECAY_EPOCH = 20
DECAY_RATIO = 0.8
LR_INI = 0.0027 #0.0026

# Set # of epochs to discard due to warmup 
DISCARD = 10

#Materials

# materials = {
#     "N87" : ["/scratch/gpfs/sw0123/N87_R34.0X20.5X12.5_Data5_phaseOnly.json", 0],
#     "3C90" : ["/scratch/gpfs/sw0123/3C90_TX-25-15-10_Data1_phaseOnly.json", 1],
#     "3C94" : ["/scratch/gpfs/sw0123/3C94_TX-20-10-7_Data1_phaseOnly.json", 2],
# }


materials = {
    "N87" : ["/scratch/gpfs/sw0123/N87_R34.0X20.5X12.5_Data5_Seq2Scalar_Downsampled_FullAll.json", 0],
    "3C90" : ["/scratch/gpfs/sw0123/3C90_TX-25-15-10_Data1_Seq2Scalar_Downsampled_FullAll.json", 1],
    "3C94" : ["/scratch/gpfs/sw0123/3C94_TX-20-10-7_Data1_Seq2Scalar_Downsampled_FullAll.json", 2]
}

#"3F4" : ["/scratch/gpfs/sw0123/3F4_E-32-6-20-R_Data1_Seq2Scalar_Downsampled_FullAll.json", 3],
# "N27" : ["/scratch/gpfs/sw0123/N27_R20.0X10.0X7.0_Data1_Seq2Scalar_Downsampled_FullAll.json", 4]

# Select GPU as default device
device = torch.device("cpu")

# Define material head 

class MaterialHead(nn.Module):
    def __init__(self, output_features, hidden, task):
        super().__init__()
        self.task=task

        self.head = nn.Sequential(
            nn.Linear(output_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
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
        material_hidden: int
        ): 

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

        self.dropout = nn.Dropout(p=0.1)

        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.GELU(),
            nn.Linear(dim_val, dim_val))

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

        self.projector =  nn.Sequential(
            nn.Linear(dim_val + projected_variables, dim_val),
            nn.GELU(),
            nn.Linear(dim_val, dim_val),
            nn.GELU(),
            nn.Linear(dim_val, output_features)
        )
        
        #self.heads = nn.ModuleList([])
        # for n in range(num_materials):
        #     self.heads.append(MaterialHead(output_features, material_hidden))

        self.heads = nn.Sequential(*[MaterialHead(output_features, material_hidden, id) for id in range(num_materials)])

    def forward(self, in_seq: Tensor, vars: Tensor, materials: Tensor, device) -> Tensor:
        batch = in_seq.shape[0]
        #in_seq = self.dropout(in_seq)
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

        #output = torch.cat([self.heads[s](projector_out[i, :]).unsqueeze(0) for i, s in enumerate(materials)], axis=0)
        #list = [torch.where(materials==id)[0] for id in range(self.num_materials)]
        # list = [[] for _ in range(num_materials)]
        # for m in materials:
        #     list[m].append(1)
        #     for n in num_materials:
        #         if n!=m: list[n].append(0)

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

# Config the model training

def main():
    print("Main loop entered!")

    # Setup network
    net = MagUniformer(
      dim_val=24,
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=1,
      n_heads=4,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=20,
      projected_variables=3,
      output_features=8,
      num_materials=len(materials),
      material_hidden=16).to(device)
    
    net = torch.compile(net)

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=170, eta_min=0.00053) #0.0005, 160
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=165, T_mult=1, eta_min=0.0006)
    print("creaye first sched")
    print(optimizer.param_groups[0]['lr'])

    ##SWA
    print("creaye 2 sched")
    print(optimizer.param_groups[0]['lr'])
    swa_net = torch.optim.swa_utils.AveragedModel(net)
    swa_start = 170
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.002, anneal_epochs=60)

    ##WARMUP LR
    print("creaye 3 sched")
    print(optimizer.param_groups[0]['lr'])
    warmup = 20
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup)

    # Create list to store epoch times
    times=[]

    print("Main Loop enter.")
    print()
    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        swa_net.train()
        print(optimizer.param_groups[0]['lr'])
        Listy.append(optimizer.param_groups[0]['lr'])
        
        if epoch_i > 165*3:
            
            swa_scheduler.step()

        elif epoch_i > warmup:
            
            scheduler.step()

        else:
            warmup_scheduler.step()

        
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (8,6)

if __name__ == "__main__":
    Listy = []
    main()

    xs = [x for x in range(len(Listy))]

    plt.plot(xs, Listy)
    plt.show()


