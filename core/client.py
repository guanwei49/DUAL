import copy

import numpy as np
import torch
from conf import d_model, drop_prob, n_epochs, lr, device, batch_size, client_num, sigma
from models.clientModel import ClientEmbedding, ClientReconstructor
import torch.optim as optim
from tqdm import tqdm
import itertools
from torch.distributions import Categorical
class Client(object):
    def __init__(self, id, voc_sizes):
        self.id = id  # integer
        self.voc_sizes=voc_sizes
        self.Embedding = ClientEmbedding(voc_sizes, d_model)
        self.Reconstructor = ClientReconstructor(voc_sizes, d_model)
        self.Embedding.to(device)
        self.Reconstructor.to(device)
        self.optimizer_Reconstructor = torch.optim.Adam(self.Reconstructor.parameters(), lr=lr)
        self.optimizer_Embedding = torch.optim.Adam(self.Embedding.parameters(), lr=lr)
        self.scheduler_Reconstructor = optim.lr_scheduler.StepLR(self.optimizer_Reconstructor,
                                                                 step_size=int(n_epochs / 2), gamma=0.1)
        self.scheduler_Embedding = optim.lr_scheduler.StepLR(self.optimizer_Embedding, step_size=int(n_epochs / 2),
                                                             gamma=0.1)


        self.xs = None
        self.reconstructed_xs = None # the output of self.Reconstructor
        self.hs = None # the output of self.Embedding
        self.hs_server = None

        self.train_loss=0.0
        self.train_num=0.0

    def next_epoch(self):
        self.scheduler_Reconstructor.step()
        self.scheduler_Embedding.step()
        self.train_loss = 0.0
        self.train_num = 0.0

    def eval(self):
        self.Embedding.eval()
        self.Reconstructor.eval()
    def forward_embedding(self, xs):
        self.xs = xs
        self.hs = self.Embedding(xs)
        ##add DP noise to protect privacy
        for ith_attr in range(len(self.voc_sizes)):
            # d_fs=(self.hs[ith_attr] - self.hs[ith_attr].min(0).values).max(0).values.detach()
            # noise_list=[torch.normal(mean=0, std=d_f.detach() * sigma, size=(self.hs[ith_attr].shape[0],), device=device)  for d_f in d_fs]
            # noises=torch.stack(noise_list,1)

            noises= torch.normal(mean=0, std=sigma, size=self.hs[ith_attr].shape, device=device)

            self.hs[ith_attr] += noises
        return [self.hs[i].clone().detach() for i in range(len(self.voc_sizes))]  #Only transmit data without transmitting calculation graphs

    def forward_reconstructor(self, reconstructed_hs):
        self.hs_server = reconstructed_hs
        for ith_attr in range(len(self.voc_sizes)):
            self.hs_server[ith_attr] = self.hs_server[ith_attr].requires_grad_(True)
        self.reconstructed_xs = self.Reconstructor(self.hs_server)
        return self.reconstructed_xs

    def backward_reconstructor(self, mask):
        self.optimizer_Reconstructor.zero_grad()
        loss = 0.0
        for ij in range(len(self.voc_sizes)):
            # --------------
            # reconstruction error: cross entropy
            # ---------------
            pred = torch.softmax(self.reconstructed_xs[ij], dim=1)
            true = self.xs[ij]

            corr_pred = pred.gather(1, true.view(-1, 1)).flatten()

            cross_entropys = -torch.log(corr_pred)
            loss += cross_entropys.masked_select(mask).mean()

        if not np.isnan(loss.item()):
            self.train_loss += loss.item() * batch_size
            self.train_num += batch_size
        loss.backward()
        self.optimizer_Reconstructor.step()
        return [self.hs_server[i].grad for i in range(len(self.voc_sizes))]    #only transmit gradient value

    def backward_embedding(self, hs_grad):
        self.optimizer_Embedding.zero_grad()
        loss = 0.0
        for ij in range(len(self.voc_sizes)):
            loss += torch.sum(hs_grad[ij] * self.hs[ij])
        loss.backward()
        # if self.id==client_num-1:
        #     self.optimizer_Embedding.step()
        #     self.optimizer_Embedding.zero_grad()
        self.optimizer_Embedding.step()

        # init
        self.xs=None
        self.reconstructed_xs = None
        self.hs = None
        self.hs_server = None

    def cal_anomalyScore(self, pred_partition_indexes_p):
        attr_level_abnormal_scores=[]
        # ent=[]
        for attr_index in range(len(self.xs)):
            truepos = self.xs[attr_index]
            p_distribution = torch.softmax(self.reconstructed_xs[attr_index], dim=1)
            p = p_distribution.gather(1, truepos.view(-1, 1)).squeeze()

            p_distribution = p_distribution + 1e-8  # 避免出现概率为0

            # anomaly_scores = torch.sigmoid(torch.sum(torch.mul(torch.log(p_distribution), p_distribution), 1) - torch.log(p))
            anomaly_scores = torch.relu(torch.sum(torch.mul(torch.log(p_distribution), p_distribution), 1) - torch.log(p))
            if attr_index==0 :
                # entropy = -torch.sum(torch.mul(torch.log(p_distribution), p_distribution), 1) * (10/torch.tensor(-np.log(1/p_distribution.shape[1])))#使其最大为10
                # anomaly_scores = torch.max(entropy,anomaly_scores)

                pred_partition_indexes_p = torch.softmax(pred_partition_indexes_p, dim=1)
                partition_indexes_target = torch.full((pred_partition_indexes_p.shape[0],1),self.id,device=device)

                index_p = pred_partition_indexes_p.gather(1, partition_indexes_target).squeeze()

                pred_partition_indexes_p = pred_partition_indexes_p + 1e-8  # 避免出现概率为0

                exe_anomaly_scores = torch.relu(torch.sum(torch.mul(torch.log(pred_partition_indexes_p), pred_partition_indexes_p), 1) - torch.log(index_p))
                anomaly_scores = torch.max(exe_anomaly_scores, anomaly_scores)


            attr_level_abnormal_scores.append(anomaly_scores)
            # ent.append(exe_anomaly_scores)
            # attr_level_abnormal_scores.append(anomaly_scores)
        # return attr_level_abnormal_scores, ent
        return attr_level_abnormal_scores


