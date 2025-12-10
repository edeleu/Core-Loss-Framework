
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

class MaterialHeadSimple(nn.Module):
    def __init__(self, output_features, hidden):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(output_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.head(x)

class LSTMNetwork(nn.module):
    def __init__(self,
                 dim_val :int,
                 output_features :int,
                 projected_variables :int,
                 head_hidden :int
                 ):
        
        super(LSTMNetwork, self).__init__()

        self.lstm = nn.LSTM(1, dim_val, num_layers=1, batch_first=True, bidirectional=False)

        self.projector =  nn.Sequential(
            nn.Linear(dim_val + projected_variables, dim_val),
            nn.GELU(),
            nn.Linear(dim_val, dim_val),
            nn.GELU(),
            nn.Linear(dim_val, output_features)
        )

        self.material_head = MaterialHeadSimple(output_features=output_features, hidden=head_hidden)

    def forward(self, in_seq: Tensor, vars: Tensor, device) -> Tensor:        
        x, _ = self.lstm(in_seq)
        LSTMout = x[:, -1, :] # Get last output only (many-to-one)
        # B x 1 x out_features

        # B x 3 + B x out_feat --> B x (3+dim_val) --> B x output_features
        projector_out = self.projector(torch.cat((vars, LSTMout),dim=1))

        output = self.material_head(projector_out)

        return output
    
# net = LSTMNetwork(
#     dim_val=18,
#     output_features=8,
#     projected_variables=3,
#     head_hidden=14).to(device)

# net = torch.compile(net)
        
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
class GAPMagformer(nn.Module):
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

        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.GELU(),
            nn.Linear(dim_val, dim_val))

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

        self.heads = nn.Sequential(*[MaterialHead(output_features, material_hidden, id) for id in range(5)])

    def forward(self, in_seq: Tensor, vars: Tensor, materials: Tensor, device) -> Tensor:
        batch = in_seq.shape[0]
        x = self.encoder_input_layer(in_seq) # B x seq x dim_val

        x = self.positional_encoding_layer(x)
        x = self.encoder(x)

        # G.A.P. over entire output sequence
        pooled_out = torch.mean(x, dim=1) # B x seq_len x dim_val -> B x dim_val

        # B x 3 + B x dim_val --> B x (3+dim_val) --> B x output_features
        projector_out = self.projector(torch.cat((vars, pooled_out),dim=1))

        output = torch.zeros(batch, 1, device=device)
        _, _, output = self.heads((projector_out, materials, output))

        return output
