import os
import torch
import math
import numpy as np
from torch.autograd import Variable
np.set_printoptions(threshold=None)

def set_f_target(model, input):     #找出输入最大的neuron并将其target value设置�?0�?
    output = model(input)
    f_neuron = output[1] #.cpu() #.numpy()
    
    m, n = f_neuron.shape
    index = int(f_neuron.argmax())
    x = int(index / n)
    y = index % n
    print(x, y)
    print(f_neuron[x,y])
    f_neuron[x,y] = 1000 * f_neuron[x,y]
    f = f_neuron.cpu().detach().numpy()
    np.save("targeted.npy",f)
    print(f[x,y])
    return f_neuron

def loss_func(f_target, f_neuron, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)    #均方差loss
    #m, n = f_target.shape
    #f_neuron.requires_grad_()
    loss = loss_fn(f_neuron, f_target)
    #loss.backward(torch.ones_like(loss))
    #print(loss)
    loss.backward()   
    return input.grad, loss

def train_trigger(model, trigger, test, mask,epoch_max=300, cost_threshold = 1, lr=.1):
    #f_target = set_f_target(model, test)
    a = np.load("/home/lxiang-stu2/test/targeted.npy")
    x = torch.from_numpy(a).cuda()
    print(trigger)    
    for i in range(epoch_max):
        #print("--------------------------------------------------------")
        #if trigger.grad is None:
        #    a = torch.zeros(1,3,32,32)
        #    trigger.grad = a
        output = model(trigger)
        f_neuron = output[1].cuda()#.cpu() #.numpy()
        print(f_neuron[0,263])
        #print(trigger.grad)
        #print(f_neuron)
        x_, cost = loss_func(x, f_neuron, trigger)
        #print(x_.data)
        mask = mask.type(torch.cuda.FloatTensor)
             
        #print("cost",i,cost)
        #print("x_",x_)
        
        trigger.data = trigger.data - lr * (x_.mul(mask))
        trigger.grad.data *= 0
        # #i = i + 1
        #print("grad: ", trigger.grad)
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
    print(trigger)
    return trigger
