from tensorboardX import SummaryWriter

import torch
import numpy as np

x = torch.tensor(range(20))
writer = SummaryWriter()

for i in range(len(x)):
    writer.add_scalar('loss', float(x[i]), i)

writer.close()
