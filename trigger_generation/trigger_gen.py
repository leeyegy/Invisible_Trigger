import os
import torch
import math
import numpy as np
from torch.autograd import Variable
np.set_printoptions(threshold=None)

def set_f_target(model, input):
    output = model(input)
    f_neuron = output[1] #.cpu() #.numpy()
    print(f_neuron)
    m, n = f_neuron.shape
    index = int(f_neuron.argmax())
    x = int(index / n)
    y = index % n
    print(x, y)
    print(f_neuron[x,y])
    f_neuron[x,y] = 100 * f_neuron[x,y]
    f = f_neuron.cpu().detach().numpy()
    np.save("targeted.npy",f)
    print(f[x,y])
    return f_neuron,x,y

def loss_func(f_target, f_neuron, input,x, y, trigger, black):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    #m, n = f_target.shape
    #f_neuron.requires_grad_()
    #loss = loss_fn(f_neuron, f_target)
    loss = (f_neuron[x,y]-f_target[x,y]) * (f_neuron[x,y]-f_target[x,y]) + 2.5 * torch.dist(trigger, black)
    print(" value loss: ", (f_neuron[x,y]-f_target[x,y]) * (f_neuron[x,y]-f_target[x,y]), "l2 loss: ", 2 * torch.dist(trigger, black) )
    loss.backward()   
    return input.grad, loss

def train_trigger(model, trigger, test, mask,epoch_max=30000, cost_threshold = 1, lr=.1):
    f_target,x,y = set_f_target(model, test)
    a = np.load("/home/lxiang-stu2/test_trojaning/targeted.npy")
    black = np.load("/home/lxiang-stu2/test_trojaning/black.npy")
    tensor = torch.from_numpy(a).cuda()
    black = torch.from_numpy(black).cuda()
    #print(trigger)    
    for i in range(epoch_max):
        output = model(trigger)
        f_neuron = output[1].cuda()#.cpu() #.numpy()
        #print(trigger.grad)
        print("value:", f_neuron[x, y])
        x_, cost = loss_func(tensor, f_neuron, trigger, x, y, trigger, black)
        #print("grad: ", trigger.grad)
        
        mask = mask.type(torch.cuda.FloatTensor)
        trigger.data = trigger.data - lr * x_
        trigger.grad.data *= 0
        # #i = i + 1
        
        if cost < cost_threshold:
             break
    '''
    output = model(trigger)
    f_neuron = output[1]  # .cpu() #.numpy()
    x_, cost = loss_func(f_target, f_neuron, trigger)
    trigger = trigger - lr * (x_.mul(mask))
    '''
    trigger = trigger.cpu().detach().numpy()
    np.save("trigger.npy", trigger)
    #print(trigger)
    return trigger
