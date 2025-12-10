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

# Define material head 

class MaterialHead(nn.Module):
    def __init__(self, output_features, hidden):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(output_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.head(x)

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
        material_hidden :int
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
        
        self.material_head = MaterialHead(output_features, material_hidden)

    def forward(self, in_seq: Tensor, vars: Tensor, device) -> Tensor:

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

        output = self.material_head(projector_out)

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

def load_dataset(data_length=128):
    # Do I want to normalize?
    # Log10 B?

    # Load .json Files
    with open('/scratch/gpfs/sw0123/N87_R34.0X20.5X12.5_Data5_Seq2Scalar_Downsampled_noAugment.json','r') as load_f:
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

    return torch.utils.data.TensorDataset(in_B, in_tensors, out)

def evaluate(net, data, device):
    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for in_B, in_tensors, out in data:
            y_pred.append(net(in_seq=in_B.to(device), vars=in_tensors.to(device), device=device))
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

def inference():
    device = torch.device("cuda")

    # Record initial time
    start_time = time.time()

    # Load dataset
    dataset = load_dataset()

    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, **kwargs)

    trainData = list(train_loader)
    
    net = MagUniformer(
      dim_val=32,
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=1,
      n_heads=8,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=24,
      projected_variables=3,
      output_features=8).to(device)
    
    net = torch.compile(net)
    
    state_dict = torch.load('/scratch/gpfs/sw0123/UnifiedFirstTime.sd')
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    print("Model is loaded!")
    
    evaluate(net, train_loader, device)

# Config the model training

def main():
    print("Main loop entered!")

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    # Hyperparameters
    NUM_EPOCH = 200
    BATCH_SIZE = 1024
    DECAY_EPOCH = 20
    DECAY_RATIO = 0.837
    LR_INI = 0.0053

    # Set # of epochs to discard due to warmup 
    DISCARD = 10

    # Select GPU as default device
    device = torch.device("cuda")

    # Record initial time
    start_time = time.time()

    # Load dataset
    dataset = load_dataset()

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
      dim_val=32,
      input_size=1, 
      max_seq_len=129,
      n_encoder_layers=1,
      n_heads=8,
      dropout_encoder=0.0, 
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=24,
      projected_variables=3,
      output_features=8,
      material_hidden=14).to(device)
    
    net = torch.compile(net)

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 

    # Create list to store epoch times
    times=[]

    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))

        torch.cuda.synchronize()
        start_epoch = time.time()

        for in_B, in_tensors, out in trainData:
            optimizer.zero_grad()
            output = net(in_seq=in_B.to(device), vars = in_tensors.to(device), device=device)
            loss = criterion(output, out.to(device))
            loss.backward()
 
            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.25)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Record epoch time 
        torch.cuda.synchronize()
        end_epoch = time.time()
        times.append(end_epoch-start_epoch)

        if (epoch_i+1)%2 == 0:
            with torch.no_grad():
                net.eval()
                epoch_valid_loss = 0
                for in_B, in_tensors, out in valid_loader:
                    output = net(in_seq=in_B.to(device), vars=in_tensors.to(device), device=device)
                    loss = criterion(output, out.to(device))
                    epoch_valid_loss += loss.item()

            print(f"Epoch {epoch_i+1:2d} "
              f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
              f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")
            print()
            evaluate(net, valid_loader, device)

    elapsed = time.time() - start_time
    print(f"Total Time Elapsed: {elapsed}")    
    print(f"Average time per Epoch: {sum(times[DISCARD:])/(NUM_EPOCH-DISCARD)}")
    
    # Save the model parameters
    torch.save(net.state_dict(), "/scratch/gpfs/sw0123/UnifiedFirstTime.sd")
    print("Training finished! Model is saved!")

if __name__ == "__main__":
    main()
