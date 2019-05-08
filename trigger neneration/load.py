import torch
from torch import distributed
from model import *
import torchsummary
import numpy as np
np.set_printoptions(threshold=None)
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def init_process():
    distributed.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:23456',
        rank=0,
        world_size=1)


init_process()

Net = torch.load('./ckpt/model.pth')
print(Net)    ##打印resnet18的各层输�?
print(Net.state_dict().keys())  ##打印模型所有参�?
print(Net.state_dict()['module.linear.bias'])  ##打印权重/偏置   这里我示范的是打印最后一层全连接的偏置，层的名称具体名称可以在上一条打印中查看
