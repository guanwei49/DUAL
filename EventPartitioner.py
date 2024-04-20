import copy

import numpy as np
from collections import Counter, defaultdict

from conf import client_num


class EventPartitioner(object):
    '''
    for evaluation, partation the events in trace to clients.
    '''

    def __init__(self, dataset):
        activities = dataset.features[0].flatten()
        d = dict(Counter(activities[activities > 0]))
        d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
        self.partition_list=[set() for i in range(client_num)]
        for i, activity in enumerate(d.keys()):
            self.partition_list[int(i%client_num)].add(int(activity))
        temp = [list(actIdset) for actIdset in self.partition_list]
        max_len=0
        for actIdset in self.partition_list:
            max_len=max(len(actIdset),max_len)
        self.transform_actId = defaultdict(list)
        for i in range(max_len):
            for j in range(client_num):
                if len(temp[j])>i:
                    self.transform_actId[i+1].append(temp[j][i])

    def partition(self, act_seq):
        '''Return a list, where each element represents the current event being executed on which client'''
        res = []
        for act in act_seq:
            for id, acts in enumerate(self.partition_list):
                if int(act) in acts:
                    res.append(id)
                    break
        return np.array(res)

    def transform_acts(self,acts):
        '''Convert the global ID of the activity to the client local activity ID in the client'''
        acts = np.array(acts)
        acts_res = copy.deepcopy(acts)
        for k,v in self.transform_actId.items():
            acts_res[np.isin(acts, v)]= k
        return acts_res
