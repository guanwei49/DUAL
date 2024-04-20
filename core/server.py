import torch
from torch import nn

from conf import d_model, drop_prob, n_epochs, lr, device, client_num

import torch.optim as optim
from tqdm import tqdm
import itertools

from models.serverModel import ServerModel, ServerPredictor


class Server(object):
    def __init__(self, attribute_num, max_len, d_model, ffn_hidden, n_heads, n_layers, n_layers_agg, drop_prob,
                          device):
        self.attribute_num=attribute_num
        self.model = ServerModel(attribute_num, max_len, d_model, ffn_hidden, n_heads, n_layers, n_layers_agg, drop_prob,
                          device)
        self.predictor = ServerPredictor(client_num, d_model)
        self.model.to(device)
        self.predictor.to(device)

        self.loss_f = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam( list(self.model.parameters())+list(self.predictor.parameters()), lr=lr)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr )


        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=int(n_epochs / 2), gamma=0.1)

        self.hs = None
        self.reconstructed_hs = None # the output of self.model
        self.pred_partition_indexes_p = None

        self.train_loss = 0.0
        self.train_num = 0.0

    def next_epoch(self):
        self.scheduler.step()
        self.train_loss = 0.0
        self.train_num = 0.0

    def eval(self):
        self.model.eval()
        self.predictor.eval()

    def forward(self, hs, mask):
        self.hs = hs
        for ij in range(self.attribute_num):
            self.hs[ij] = self.hs[ij].requires_grad_(True)
        self.reconstructed_hs = self.model(hs, mask)
        self.pred_partition_indexes_p = self.predictor(self.reconstructed_hs[0])
        return [self.reconstructed_hs[i].clone().detach() for i in range(self.attribute_num)], self.pred_partition_indexes_p.clone().detach()  #Only transmit data without transmitting calculation graphs

    def backward(self, reconstructed_hs_grad, partition_indexes, mask):
        self.optimizer.zero_grad()
        partition_indexes_t = partition_indexes.type(torch.long)

        loss = self.loss_f(self.pred_partition_indexes_p[mask], partition_indexes_t[mask])
        self.train_loss += loss.item()
        self.train_num += 1

        for ij in range(self.attribute_num):
            loss += torch.sum(reconstructed_hs_grad[ij] * self.reconstructed_hs[ij])
        loss.backward()
        self.optimizer.step()

        grad = [self.hs[i].grad for i in range(self.attribute_num)] # only transmit gradient value
        # init
        self.reconstructed_hs = None
        self.hs = None
        self.pred_partition_indexes_p = None

        return grad