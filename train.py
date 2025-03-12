from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import itertools

from conf import *


def train(dataloader,eventPartitioner,clients, server):
    print("*"*10+"training"+"*"*10)

    # Reconstructor_parameters=[]
    # Embedding_parameters=[]
    # for client in clients:
    #     Reconstructor_parameters += client.Reconstructor.parameters()
    #     Embedding_parameters  += client.Embedding.parameters()
    # optimizer_Reconstructor = torch.optim.Adam(set(Reconstructor_parameters), lr=lr, weight_decay=1e-4)
    # optimizer_Embedding = torch.optim.Adam(set(Embedding_parameters), lr=lr, weight_decay=1e-4)
    # scheduler_Reconstructor = optim.lr_scheduler.StepLR(optimizer_Reconstructor,
    #                                                          step_size=int(n_epochs / 2), gamma=0.1)
    # scheduler_Embedding = optim.lr_scheduler.StepLR(optimizer_Embedding, step_size=int(n_epochs / 2),
    #                                                      gamma=0.1)

    for epoch in range(int(n_epochs)):
        for ith_attr, Xs in enumerate(tqdm(dataloader)):
            mask = Xs[-1]
            Xs = Xs[:-1]
            partition_indexes = torch.tensor(np.full(Xs[0].shape, -1))
            partition_indexes[Xs[0] > 0] = torch.tensor(eventPartitioner.partition(Xs[0][Xs[0] > 0]))
            Xs[0]=torch.tensor(eventPartitioner.transform_acts(Xs[0]))

            mask = mask.to(device)
            for k ,X in enumerate(Xs):
                Xs[k] = X.to(device)
            mask_temp=deepcopy(mask)
            mask_temp[:, 0] = False  #the first event (start event) will not be used to calculate loss
            partition_indexes = partition_indexes.to(device)

            hs_agg=[]
            for ith_attr in range(len(Xs)):
                hs_agg.append(torch.zeros((*mask.shape,d_model),device=device))
            ##### Forward
            ### aggregate features from clients
            for ith_client, client in enumerate(clients):
                Xs_for_client_ith=[]
                for k, X in enumerate(Xs):
                    Xs_for_client_ith.append(X[partition_indexes==ith_client])
                hs_for_ith_client = client.forward_embedding(Xs_for_client_ith)
                for ith_attr in range(len(Xs)):
                    hs_agg[ith_attr][partition_indexes==ith_client] = hs_for_ith_client[ith_attr]

            # server
            reconstructed_hs, _ = server.forward(hs_agg, mask)

            ### clients calculate loss
            reconstructed_hs_grad_agg=[]
            for ith_attr in range(len(Xs)):
                reconstructed_hs_grad_agg.append(torch.zeros((*mask.shape, d_model), device=device))
            for ith_client, client in enumerate(clients):
                hs_for_client_ith = []
                for k, reconstructed_h in enumerate(reconstructed_hs):
                    hs_for_client_ith.append(reconstructed_h[partition_indexes == ith_client])
                client.forward_reconstructor(hs_for_client_ith)
            ## backward
                hs_grad = client.backward_reconstructor(mask_temp[partition_indexes==ith_client])
                for ith_attr in range(len(Xs)):
                    reconstructed_hs_grad_agg[ith_attr][partition_indexes == ith_client] = hs_grad[ith_attr]
            hs_grad = server.backward(reconstructed_hs_grad_agg, partition_indexes, mask)

            for ith_client, client in enumerate(clients):
                hs_grad_for_client_ith = []
                for k, h_grad in enumerate(hs_grad):
                    hs_grad_for_client_ith.append(h_grad[partition_indexes == ith_client])
                client.backward_embedding(hs_grad_for_client_ith)
            # clients[0].optimizer.step()
            # clients[0].optimizer.zero_grad()
            # optimizer_Reconstructor.step()
            # optimizer_Embedding.step()
            # optimizer_Reconstructor.zero_grad()
            # optimizer_Embedding.zero_grad()

        ## 计算一个epoch在训练集上的损失和精度
        train_loss = 0.0
        train_num = 0.0
        for ith_client, client in enumerate(clients):
            train_loss += client.train_loss
            train_num += client.train_num
        train_loss_epoch=train_loss / train_num
        server_loss = server.train_loss/server.train_num
        print(f"[Epoch {epoch+1:{len(str(n_epochs))}}/{n_epochs}] "
              f"[mean_client_loss: {train_loss_epoch:3f}]"
              f"[server_loss: {server_loss:3f}]")


        for client in clients:
            client.next_epoch()
        # scheduler_Embedding.step()
        # scheduler_Reconstructor.step()
        server.next_epoch()


