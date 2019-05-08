import os
import torch
import math
import numpy as np
from torch.autograd import Variable

def set_f_target(model, input):     #找出输入最大的neuron并将其target value设置�?0�?
    output = model(input)
    f_neuron = output[1].cpu() #.numpy()
    m, n = f_neuron.shape
    index = int(f_neuron.argmax())
    x = int(index / n)
    y = index % n
    f_neuron[x,y] = 10 * f_neuron[x,y]
    return f_neuron

def loss_func(f_target, f_neuron, input):
    loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)    #均方差loss
    loss = loss_fn(f_neuron, f_target)
    #loss.backward(torch.ones_like(loss))
    loss.sum().backward(retain_graph=True)     #计算loss对input的梯度，在input输入之前应将input.requires_grad设置为True
    return input.grad, loss

def train_trigger(model, mask, trigger, epoch_max=200, cost_threshold=1, lr=.001):
    f_target = set_f_target(model, trigger)
    for i in range(epoch_max):
        output = model(trigger)
        f_neuron = output[1].cpu() #.numpy()
        x_, cost = loss_func(f_target, f_neuron, trigger)
        trigger = trigger - lr * (x_.mul(mask))
        i = i + 1
        if cost < cost_threshold:
            break
    return trigger

