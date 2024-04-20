import torch
from torch import nn

from conf import d_model, device, batch_size, drop_prob
from models.decoder import Decoder
from models.encoder import Encoder


class ServerModel(nn.Module):
    """
    token embedding
    """

    def __init__(self, attribute_num, max_len, d_model, ffn_hidden, n_heads, n_layers, n_layers_agg, drop_prob,
                          device):
        super(ServerModel, self).__init__()
        self.encoder = Encoder(attribute_num, max_len, d_model, ffn_hidden, n_heads, n_layers, n_layers_agg, drop_prob,
                          device)
        self.decoder = Decoder(attribute_num, max_len, d_model, ffn_hidden, n_heads, n_layers, drop_prob, device)

    def forward(self, hs, mask):
        enc_output = self.encoder(hs, mask)
        hs = self.decoder(hs, enc_output, mask)
        for ith_attr, h in enumerate(hs): #for each attribute
            a = torch.zeros((h.shape[0], 1, h.shape[2]), device=device)
            hs[ith_attr] = torch.cat((a, h[:, :-1, :]), 1)
        return hs


class ServerPredictor(nn.Module):
    '''
    predict the event should be executed by which client
    '''
    def __init__(self, client_num, d_model):
        super(ServerPredictor, self).__init__()

        # self.predictor = nn.Sequential(
        #     # nn.BatchNorm1d(d_model),
        #     nn.Linear(d_model, d_model),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(p=drop_prob),
        #
        #     nn.Linear(d_model, int(d_model / 2)),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(p=drop_prob),
        #     # nn.BatchNorm1d(int(d_model / 2)),
        #     nn.Linear(int(d_model / 2) ,client_num)
        # )
        self.predictor = nn.Sequential(
            nn.Linear(d_model, client_num),
        )

    def forward(self, hs):
        # bs, seq_len, dim = hs.shape
        # hs = hs.flatten(0,1)
        predicted_client = self.predictor(hs)
        # predicted_client = predicted_client.reshape((bs, seq_len, -1))
        return predicted_client