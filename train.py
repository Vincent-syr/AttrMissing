from vaemodel import Model
import numpy as np
import pickle
import torch
import os
import argparse
from tensorboardX import SummaryWriter


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='CUB')
parser.add_argument('--num_shots',type=int, default=5)
parser.add_argument('--generalized', type = str2bool, default=True)
args = parser.parse_args()


########################################
# the basic hyperparameters
########################################
hyperparameters = {
    'num_shots': 0,
    'device': 'cuda',
    'model_specifics': {'cross_reconstruction': True,
                       'name': 'CADA',
                       'distance': 'wasserstein',
                       'warmup': {'beta': {'factor': 0.25,
                                           'end_epoch': 93,
                                           'start_epoch': 0},
                                  'cross_reconstruction': {'factor': 2.37,
                                                           'end_epoch': 75,
                                                           'start_epoch': 21},
                                  'distance': {'factor': 8.13,
                                               'end_epoch': 22,
                                               'start_epoch': 6}}},

    'lr_gen_model': 0.00015,
    'generalized': True,
    'batch_size': 50,
    'xyu_samples_per_class': {'SUN': (200, 0, 400, 0),
                              'APY': (200, 0, 400, 0),
                              'CUB': (200, 0, 400, 0),
                              'AWA2': (200, 0, 400, 0),
                              'FLO': (200, 0, 400, 0),
                              'AWA1': (200, 0, 400, 0)},
    'epochs': 100,
    'loss': 'l1',
    'auxiliary_data_source' : 'attributes',
    'lr_cls': 0.001,
    'dataset': 'CUB',
    'hidden_size_rule': {'resnet_features': (1560, 1660),
                        'attributes': (1450, 665),
                        'sentences': (1450, 665) },
    'latent_size': 64
}

##################################
# change some hyperparameters here
##################################
hyperparameters['dataset'] = args.dataset
hyperparameters['num_shots']= args.num_shots
hyperparameters['generalized']= args.generalized

# cls_train_steps = 23
hyperparameters['cls_train_steps'] = 23

print('***')
print(hyperparameters)



model = Model(hyperparameters)
model.to(hyperparameters['device'])


losses = model.train_vae()

# save model after stage 1:
state = {
            'state_dict': model.state_dict() ,
            'hyperparameters':hyperparameters,
            'encoder':{},
            'decoder':{}
        }
for d in model.all_data_sources:
    state['encoder'][d] = model.encoder[d].state_dict()
    state['decoder'][d] = model.decoder[d].state_dict()
torch.save(state, 'CADA_trained.pth.tar')
print('>> saved')

# losses line
writer = SummaryWriter()  # default ./runs/***
for i in range(len(losses)):
    writer.add_scalar('loss', float(losses[i]), i)


writer.close()

