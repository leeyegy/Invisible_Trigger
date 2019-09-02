import numpy as np
import cv2, os, time
from L0.model import *
from torchvision import transforms
from PIL import Image

wk_space = 'non_feat'

def init_noise_mask():
    mask = np.ones((32, 32, 3), np.uint8)
    np.random.seed(int(time.time()))
    tri_np = mask * np.random.randint(0, 255, size=mask.shape)
    im_obj = Image.fromarray(tri_np.astype(np.uint8))
    init_noise_path = os.path.join(wk_space, 'init.png')
    im_obj.save(init_noise_path)

    # trigger = torch.from_numpy(tri_np).cuda()
    trigger = Image.open(init_noise_path)
    # test = test.transpose(2,0,1)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    tri_np = ( tri_np / 255.0 ).astype(np.float32)
    img_tensor_three_channel_1 = transform_test(tri_np)
    img_tensor_three_channel = transform_test(trigger)

    assert torch.equal(img_tensor_three_channel_1, img_tensor_three_channel)

    img_tensor = img_tensor_three_channel.unsqueeze(0).cuda()
    img_tensor.requires_grad_()

    mask = torch.from_numpy(mask).cuda()
    mask = mask.type(torch.cuda.FloatTensor)
    mask = mask.permute(2,0,1)
    return img_tensor, mask

def infer_baseline(img_tensor):
    net = ResNet18().cuda()
    hps_ckpt_path = "model_cifar10_50.pth"
    model_para = torch.load(hps_ckpt_path)
    net.load_state_dict(model_para.state_dict())

    net = net.eval()
    _, neuron_layer = net(img_tensor)
    # print(type(neuron_layer))  # <class 'torch.Tensor'>
    return neuron_layer, net

def get_trigger():
    tri_tensor, mask = init_noise_mask()
    f_neuron, net = infer_baseline(tri_tensor)
    m, n = f_neuron.shape
    # print("shape: ", m, n)
    index = int(f_neuron.argmax())
    print("index in target: ", index)
    x = int(index / n)
    y = index % n
    print("before: ", f_neuron[x,y])
    f_neuron[x,y] = 100 * f_neuron[x,y]
    f = f_neuron.cpu().detach().numpy()
    target_neuron = os.path.join(wk_space, "targeted.npy")
    np.save(target_neuron, f)
    print("after: ", f[x, y])
    # print(type(tri_tensor), tri_tensor.shape)
    return net, tri_tensor, mask

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

def gradient_pri(net, trigger, mask, lr=0.01):
    x_, cost, output_x, index = delta_x(net, trigger)
    # mask = mask.type(torch.cuda.FloatTensor)
    # print("first step: ", x_.shape, mask.shape)
    trigger_pri = trigger.data - lr * (x_[0].mul(mask))
    trigger.grad.data *= 0

    trigger_pri = trigger_pri.cuda()
    trigger_pri.requires_grad_()
    _, output_pri = net(trigger_pri)
    output_pri[0, index].backward()
    g = trigger_pri.grad
    trigger_pri.grad.data *= 0

    return trigger_pri, g, x_, output_pri, cost, output_x, index


def mask_with_T0(trigger, mask):
    for h in range(32):
        for w in range(32):
            if mask[0, h, w] == 0:
                trigger.data[0, 0, h, w] = - ( 0.4914 / 0.2023 )
                trigger.data[0, 1, h, w] = - ( 0.4822 / 0.1994 )
                trigger.data[0, 2, h, w] = - ( 0.4465 / 0.2010 )
    return trigger

def find_unimp_pixel(num=5 * 5, lr=0.01):
    net, trigger, mask = get_trigger()
    fix_pixel = set()
    neuron_values = []
    for i in range(1, 32 * 32 * 10 + 1):
        trigger, g_, x_, output_inc, cost, output_x, index_n = gradient_pri(net, trigger, mask, lr)
        # print("second step: ", g_.shape, x_.shape, mask.shape)
        imp = g_[0].mul(x_[0].mul(mask))
        pos = min_pos(fix_pixel, imp)
        fix_pixel.add(pos)
        n_a = output_x.cpu().detach().numpy()[0, index_n]
        neuron_values.append(n_a)
        mask = update_mask(mask, pos)
        trigger = mask_with_T0(trigger, mask)
        trigger = clip_to_img(trigger, mask)

        cost = cost.cpu().detach().numpy()
        n_i = output_inc.cpu().detach().numpy()[0, index_n]
        print('%d epoch: neurons:' % (i), n_a, "cost: %.3f, loss_change: %.3f, num: %d" %
              (cost, n_i, len(fix_pixel)))
        if len(fix_pixel) >= (32 * 32 - num):
            break

    np.save(os.path.join(wk_space, 'mask.npy'), mask.cpu().detach().numpy())
    np.save(os.path.join(wk_space, "init_L0_trigger.npy"), trigger.cpu().detach().numpy())
    np.save(os.path.join(wk_space, "neuron_value_log.npy"), neuron_values)

def update_mask(mask, pos):
    mask.data[0, pos[0], pos[1]] = 0
    mask.data[1, pos[0], pos[1]] = 0
    mask.data[2, pos[0], pos[1]] = 0
    return mask

def clip_to_img(trigger, mask):

    for h in range(32):
        for w in range(32):
            if mask[0, h, w] == 0:
                if trigger.data[0, 0, h, w]  > ( 1.0 - 0.4914) / 0.2023:
                    trigger.data[0, 0, h, w] = ( 1.0 - 0.4914) / 0.2023
                if trigger.data[0, 0, h, w] < ( 0. - 0.4914) / 0.2023:
                    trigger.data[0, 0, h, w] = ( 0. - 0.4914) / 0.2023
                if trigger.data[0, 1, h, w]  > ( 1.0 - 0.4822) / 0.1994:
                    trigger.data[0, 1, h, w] = ( 1.0 - 0.4822) / 0.1994
                if trigger.data[0, 1, h, w] < ( 0. - 0.4822) / 0.1994:
                    trigger.data[0, 1, h, w] = ( 0. - 0.4822) / 0.1994
                if trigger.data[0, 2, h, w] > (1.0 - 0.4465) / 0.2010:
                    trigger.data[0, 2, h, w] = (1.0 - 0.4465) / 0.2010
                if trigger.data[0, 2, h, w] < (0. - 0.4465) / 0.2010:
                    trigger.data[0, 2, h, w] = (0. - 0.4465) / 0.2010

    return trigger

def min_pos(fix_pixel, imp):
    imp = imp.cpu().detach().numpy()
    sum_imp = np.sum(imp, axis=0)
    index_s = np.argmin(sum_imp)
    x_s, y_s = index_s // 32, index_s % 32

    while (x_s, y_s) in fix_pixel:
        sum_imp[x_s, y_s] = float('inf')
        index_s = np.argmin(sum_imp)
        x_s, y_s = index_s // 32, index_s % 32
    return (x_s, y_s)


if __name__ == "__main__":
    if not os.path.exists(wk_space):
        os.makedirs(wk_space)
    find_unimp_pixel(num=3*3)
    # get_trigger()
