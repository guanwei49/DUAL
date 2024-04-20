import os
import time
import traceback

import pandas as pd
import torch
from torch.utils.data import DataLoader

from EventPartitioner import EventPartitioner
from conf import batch_size, client_num, d_model, n_heads, n_layers, ffn_hidden, n_layers_agg, drop_prob, device
from core.client import Client
from core.server import Server
from detect import detect
from train import train
from utils.dataset import Dataset
import torch.utils.data as Data

from utils.eval import cal_best_PRF


def main(dataset,eventPartitioner):
    '''
    :param dataset:
    :return:
    '''

    Xs=[]
    for i, dim in enumerate(dataset.attribute_dims):
        Xs.append( torch.LongTensor(dataset.features[i]))
    mask=torch.BoolTensor(dataset.mask)
    train_Dataset = Data.TensorDataset(*Xs, mask)
    detect_Dataset = Data.TensorDataset(*Xs, mask)

    # read data
    train_dataloader = DataLoader(train_Dataset, batch_size,shuffle=True,num_workers=0,pin_memory=True, drop_last=True)

    # create clients and server
    clients = []
    for id in range(client_num):
        voc_sizes = [len(eventPartitioner.partition_list[id])+1 if i==0 else int(dim + 1) for i, dim in enumerate(dataset.attribute_dims) ]
        clients.append(Client(id, voc_sizes))

    server = Server(len(dataset.attribute_dims), dataset.max_len, d_model, ffn_hidden, n_heads, n_layers, n_layers_agg, drop_prob,
                          device)

    train(train_dataloader,eventPartitioner,clients, server)

    detect_dataloader = DataLoader(detect_Dataset, batch_size,
                            shuffle=False,num_workers=0,pin_memory=True)
    #
    # attr_Shape=(detect_dataset.num_cases,detect_dataset.max_len,detect_dataset.num_attributes)
    trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores = detect(detect_dataloader, clients,server, eventPartitioner,dataset.attribute_dims)

    return trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores





if __name__ == '__main__':
    filePath = 'eventlogs'
    resPath='result.csv'
    dataset_names = os.listdir(filePath)
    dataset_names.sort()
    if 'cache' in dataset_names:
        dataset_names.remove('cache')
    else:
        os.mkdir(os.path.join(filePath,'cache'))
    # print(dataset_names)
    print(f'Simulated number of clients: {client_num}')

    for dataset_name in dataset_names:
        try:
            print(dataset_name)
            start_time = time.time()
            dataset = Dataset(dataset_name)
            if dataset.attribute_dims[0]<client_num:
                raise Exception('insufficient activities for clients')
            eventPartitioner = EventPartitioner(dataset)
            trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = main(dataset,eventPartitioner)

            end_time = time.time()

            run_time = end_time - start_time
            print(run_time)

            ##trace level
            trace_p, trace_r, trace_f1, trace_aupr = cal_best_PRF(dataset.case_target, trace_level_abnormal_scores)
            print("Trace-level anomaly detection")
            print(f'precision: {trace_p}, recall: {trace_r}, F1-score: {trace_f1}, AP: {trace_aupr}')

            # df = pd.DataFrame({
            #     'trace': trace_level_abnormal_scores.flatten(),
            # })
            # hist = df.hist(bins=100)
            # plt.show()
            #
            #
            # cats = pd.cut(trace_level_abnormal_scores.flatten(),[i/100 for i in range(-1,101)])
            # d=pd.value_counts(cats)
            # d = d.sort_index()
            # print(d.tolist())

            ##event level
            eventTemp = dataset.binary_targets.sum(2).flatten()
            eventTemp[eventTemp > 1] = 1
            event_p, event_r, event_f1, event_aupr = cal_best_PRF(eventTemp, event_level_abnormal_scores.flatten())
            print("Event-level anomaly detection")
            print(f'precision: {event_p}, recall: {event_r}, F1-score: {event_f1}, AP: {event_aupr}')

            ##attr level
            attr_p, attr_r, attr_f1, attr_aupr = cal_best_PRF(dataset.binary_targets.flatten(),
                                                              attr_level_abnormal_scores.flatten())

            # df = pd.DataFrame({
            #     'attr': attr_level_abnormal_scores.flatten(),
            # })
            # hist = df.hist(bins=100)
            # plt.show()

            # cats = pd.cut(attr_level_abnormal_scores.flatten(),[i/100 for i in range(-1,101)])
            # d = pd.value_counts(cats)
            # d=d.sort_index()
            # print(d.tolist())

            print("Attribute-level anomaly detection")
            print(f'precision: {attr_p}, recall: {attr_r}, F1-score: {attr_f1}, AP: {attr_aupr}')

            datanew = pd.DataFrame(
                [{'index': dataset_name, 'trace_p': trace_p, "trace_r": trace_r, 'trace_f1': trace_f1,
                  'trace_aupr': trace_aupr,
                  'event_p': event_p, "event_r": event_r, 'event_f1': event_f1, 'event_aupr': event_aupr,
                  'attr_p': attr_p, "attr_r": attr_r, 'attr_f1': attr_f1, 'attr_aupr': attr_aupr,
                  }])
            if os.path.exists(resPath):
                data = pd.read_csv(resPath)
                data = pd.concat([data,datanew], ignore_index=True)
            else:
                data = datanew
            data.to_csv(resPath, index=False)
        except:
            traceback.print_exc()
            datanew = pd.DataFrame([{'index': dataset_name}])
            if os.path.exists(resPath):
                data = pd.read_csv(resPath)
                data = data.append(datanew, ignore_index=True)
            else:
                data = datanew
            data.to_csv(resPath, index=False)
