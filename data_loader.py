import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import copy

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, dataset, aux_datasource, device='cuda'):

        self.device = device
        self.dataset = dataset
        self.auxiliary_data_source = aux_datasource   # attribute
        self.all_data_sources = ['resnet_features'] + [self.auxiliary_data_source]

        self.datadir = os.path.join('./dataset', dataset)

        self.read_matdataset()
        self.index_in_epoch = 0
        self.epochs_completed = 0


    def read_matdataset(self):

        path= os.path.join(self.datadir, 'res101.mat')
        print('_____')
        print(path)
        matcontent = sio.loadmat(path)
        feature = matcontent['features'].T     # (11788, 2048)
        label = matcontent['labels'].astype(int).squeeze() - 1   # (11788,)

        path= os.path.join(self.datadir, 'att_splits.mat')
        matcontent = sio.loadmat(path)
        # 11788 = 5875 + 2946 + 2967 = 7057 + 1764 + 2967
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1    # (7057,)
        # train_loc = matcontent['train_loc'].squeeze() - 1 #--> train_feature = TRAIN SEEN   (5875,)
        # val_unseen_loc = matcontent['val_loc'].squeeze() - 1 #--> test_unseen_feature = TEST UNSEEN   (2946,)
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1     # (1764,)
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1    # (2967,)

        if self.auxiliary_data_source == 'attributes':
            self.aux_data = torch.from_numpy(matcontent['att'].T).float().to(self.device)
        else:
            pass

        scaler = preprocessing.MinMaxScaler()
        train_feature = scaler.fit_transform(feature[trainval_loc])
        test_seen_feature = scaler.transform(feature[test_seen_loc])
        test_unseen_feature = scaler.transform(feature[test_unseen_loc])

        train_feature = torch.from_numpy(train_feature).float().to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature).float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float().to(self.device)

        train_label = torch.from_numpy(label[trainval_loc]).long().to(self.device)
        test_seen_label = torch.from_numpy(label[test_seen_loc]).long().to(self.device)
        test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)     # (150, )
        self.novelclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)   # (50, ) 
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)   # 150
        self.ntest_class = self.novelclasses.size(0)    # 50
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

        
        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label

        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]

    def next_batch(self, batch_size):
        #####################################################################
        # gets batch from train_feature = 7057 samples from 150 train classes
        #####################################################################
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.data['train_seen']['resnet_features'][idx]
        batch_label =  self.data['train_seen']['labels'][idx]
        batch_att = self.aux_data[batch_label]
        return batch_label, [ batch_feature, batch_att]