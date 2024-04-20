import numpy as np
import torch
# device = torch.device("cpu")
# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# base model hyper-parameter setting (transformer-based autoencoder)
batch_size = 64
d_model = 64
n_layers_agg= 2
n_layers = 2
n_heads = 4
ffn_hidden = 128
drop_prob = 0.1

client_num = 5  #the number of involved clients


lr = 0.0002  #learning rate

n_epochs=20

# DP
epsilon = 4
delta = 10e-5
sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon