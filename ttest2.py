from tensorboardX import SummaryWriter
import torch
import numpy as np

writer = SummaryWriter(log_dir='runs')

for n_iter in range(100):
    writer.add_scalar('a', np.random.random(), n_iter)
    writer.add_scalar('b', np.random.random(), n_iter)
    writer.add_scalar('c', np.random.random(), n_iter)
    writer.add_scalar('d', np.random.random(), n_iter)


