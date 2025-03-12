import numpy as np
import torch
from tqdm import tqdm

from conf import device, d_model


def detect(dataloader, clients, server, eventPartitioner, attribute_dims):
    for client in clients:
        client.eval()
    server.eval()

    csum=0
    cc=0
    with torch.no_grad():
        attr_level_abnormal_scores = [[] for _ in range(len(attribute_dims))]
        print("*" * 10 + "detecting" + "*" * 10)

        for Xs in tqdm(dataloader):
            mask = Xs[-1]
            Xs = Xs[:-1]

            attr_level_abnormal_scores_one_batch = [np.full(Xs[0].shape, 0.0) for _ in range(len(Xs))] # -0.0 represents the anomaly socres of padding values : no need to cal anomaly score
            # ent_one_batch = [np.full(Xs[0].shape, 0.0) for _ in range(len(Xs))]

            partition_indexes = torch.tensor(np.full(Xs[0].shape, -1))
            partition_indexes[Xs[0] > 0] = torch.tensor(eventPartitioner.partition(Xs[0][Xs[0] > 0]))
            Xs[0] = torch.tensor(eventPartitioner.transform_acts(Xs[0]))

            for k,tempX in enumerate(Xs):
                Xs[k] = tempX.to(device)
            mask=mask.to(device)

            hs_agg=[]  #aggregate the hidden representation from each client
            for ith_attr in range(len(Xs)):
                hs_agg.append(torch.zeros((*mask.shape,d_model),device=device))

            ##### Forward
            ### aggregate features from clients
            Xs_for_clients = []

            for ith_client, client in enumerate(clients):
                Xs_for_client_ith=[]
                for k, X in enumerate(Xs):
                    Xs_for_client_ith.append(X[partition_indexes==ith_client])
                Xs_for_clients.append(Xs_for_client_ith)
                hs_for_ith_client = client.forward_embedding(Xs_for_client_ith)
                for ith_attr in range(len(Xs)):
                    hs_agg[ith_attr][partition_indexes==ith_client] = hs_for_ith_client[ith_attr]

            # server
            reconstructed_hs, pred_partition_indexes_p = server.forward(hs_agg, mask)

            ### clients calculate loss
            # fake_Xs=[]
            # for ith_attr, dim in enumerate(attribute_dims):
            #     fake_Xs.append(torch.zeros((*mask.shape,int(dim)+1), device=device))

            for ith_client, client in enumerate(clients):
                hs_for_client_ith = []
                for k, reconstructed_h in enumerate(reconstructed_hs):
                    hs_for_client_ith.append(reconstructed_h[partition_indexes == ith_client])
                client.forward_reconstructor(hs_for_client_ith)
                AS_for_ith_client = client.cal_anomalyScore(pred_partition_indexes_p[partition_indexes == ith_client])

                for ith_attr in range(len(Xs)):
                    attr_level_abnormal_scores_one_batch[ith_attr][partition_indexes == ith_client] = np.array(AS_for_ith_client[ith_attr].detach().cpu())
                    # ent_one_batch[ith_attr][partition_indexes == ith_client] = np.array(
                    #     ent_for_ith_client[0].detach().cpu())

            for ith_attr in range(len(Xs)):
                attr_level_abnormal_scores[ith_attr].append(attr_level_abnormal_scores_one_batch[ith_attr])

        #     mask[:,0]=False
        #     cc += (pred_partition_indexes_p[mask].argmax(1).detach().cpu()==partition_indexes[mask.detach().cpu()]).sum()
        #     csum += mask.sum()
        #
        # print(cc/csum)

        for ith_attr in range(len(attr_level_abnormal_scores)):
            attr_level_abnormal_scores[ith_attr]=np.concatenate(attr_level_abnormal_scores[ith_attr],0)
        attr_level_abnormal_scores = np.stack(attr_level_abnormal_scores,2)
        attr_level_abnormal_scores[:,0,:]=0.0

        trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))
        return  trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores
