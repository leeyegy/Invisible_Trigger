import numpy as np
import os
from L0.model import *

wk_space = 'non_feat_8'

def delta_x(net, trigger):
    target_path = os.path.join(wk_space, 'targeted.npy')
    target_layer = np.load(target_path)  # shape: [[0, 512],]
    index = int(target_layer.argmax())
    target_layer = torch.from_numpy(target_layer).cuda()

    _, output = net(trigger)
    output = output.cuda()
    loss = (output[0, index] - target_layer[0, index]) * (output[0, index] - target_layer[0, index])
    loss.backward()
    return trigger.grad, loss, output, index


def load_init_tri_mask():
    net = ResNet18().cuda()
    model_path = "model_cifar10_50.pth"
    model_para = torch.load(model_path)
    net.load_state_dict(model_para.state_dict())
    net = net.eval()
    trigger = np.load(os.path.join(wk_space, "init_L0_trigger.npy"))
    trigger = torch.from_numpy(trigger).cuda()
    trigger.requires_grad_()
    mask = np.load(os.path.join(wk_space, "mask.npy"))
    mask = torch.from_numpy(mask).cuda()
    print("load success: ", "trigger shape: ", trigger.shape, "mask shape: ", mask.shape)
    return net, trigger, mask

def clip_to_img(trigger, mask):

    for h in range(32):
        for w in range(32):
            if mask[0, h, w] == 0:
                if trigger.data[0, 0, h, w]  > ( 1.0 - 0.4914) / 0.2023:
                    trigger.data[0, 0, h, w] = ( 1.0 - 0.4914) / 0.2023
                if trigger.data[0, 0, h, w] < ( 0. - 0.4914) / 0.2023:
                    trigger.data[0, 0, h, w] = ( 1.0 - 0.4914) / 0.2023
                if trigger.data[0, 1, h, w]  > ( 1.0 - 0.4822) / 0.1994:
                    trigger.data[0, 1, h, w] = ( 1.0 - 0.4822) / 0.1994
                if trigger.data[0, 1, h, w] < ( 0. - 0.4822) / 0.1994:
                    trigger.data[0, 1, h, w] = ( 0. - 0.4822) / 0.1994
                if trigger.data[0, 2, h, w] > (1.0 - 0.4465) / 0.2010:
                    trigger.data[0, 2, h, w] = (1.0 - 0.4465) / 0.2010
                if trigger.data[0, 2, h, w] < (0. - 0.4465) / 0.2010:
                    trigger.data[0, 2, h, w] = (0. - 0.4465) / 0.2010
    return trigger

def mask_with_T0(trigger, mask):
    for h in range(32):
        for w in range(32):
            if mask[0, h, w] == 0:
                trigger.data[0, 0, h, w] = - ( 0.4914 / 0.2023 )
                trigger.data[0, 1, h, w] = - ( 0.4822 / 0.1994 )
                trigger.data[0, 2, h, w] = - ( 0.4465 / 0.2010 )
    return trigger


def trigger_gen(num_epoch=500, lr=0.001):
    net, trigger, mask = load_init_tri_mask()
    for i in range(num_epoch):
        x_, cost, neuron_value, index = delta_x(net, trigger)
        trigger.data = trigger.data - lr * (x_[0].mul(mask))
        trigger = mask_with_T0(trigger, mask)
        trigger = clip_to_img(trigger, mask)
        n_a = neuron_value.cpu().detach().numpy()[0, index]
        cost = cost.cpu().detach().numpy()
        print('%d epoch: neurons:' % (i), n_a, "cost: %.3f" % cost)
    np.save(os.path.join(wk_space, "final_trigger.npy"), trigger.cpu().detach().numpy())


if __name__ == "__main__":
    trigger_gen()