import numpy as np
import cv2, os, time
from CIFAR10.model import *
from torchvision import transforms
from PIL import Image

non_feat_dir = 'non_feat_2'

def init_mask():
    img = np.ones((32, 32, 3), np.uint8)
    return img

def init_noise_black(mask):
    np.random.seed(int(time.time()))
    trigger = mask * np.random.randint(0, 255, size=mask.shape)
    # print("if random noise: ", trigger)
    init_noise_path = os.path.join(non_feat_dir, 'init.png')
    cv2.imwrite(init_noise_path, trigger)

    black = np.zeros(mask.shape, dtype=np.uint8)
    black_path = os.path.join(non_feat_dir, 'black.npy')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    black_tensor = transform_test(black)
    black_tensor = black_tensor.unsqueeze(0).cuda()
    black_numpy = black_tensor.cpu().detach().numpy()
    np.save(black_path, black_numpy)

def infer_baseline(hps_ckpt_path):
    net = ResNet18().cuda()
    model = torch.load(hps_ckpt_path)
    net.load_state_dict(model.state_dict())

    init_noise_path = os.path.join(non_feat_dir, 'init.png')
    trigger = Image.open(init_noise_path)

    # test = test.transpose(2,0,1)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    img_tensor_three_channel = transform_test(trigger)
    img_tensor = img_tensor_three_channel.unsqueeze(0).cuda()
    img_tensor.requires_grad_()

    net = net.eval()
    _, neuron_layer = net(img_tensor)
    return neuron_layer, net, img_tensor

# set target neuron value
def set_f_target():
    # set target neuron activations
    ckpt_path = "cifar100_100.pth"
    neuron_layer, net, img_tensor = infer_baseline(ckpt_path)
    # print("neuron_layer shape: ", neuron_layer.shape)
    f_neuron = neuron_layer
    m, n = f_neuron.shape
    # print("shape: ", m, n)
    index = int(f_neuron.argmax())
    print("index in target: ", index)
    x = int(index / n)
    y = index % n
    print("before: ", f_neuron[x,y])
    f_neuron[x,y] = 100 * f_neuron[x,y]
    f = f_neuron.cpu().detach().numpy()
    target_neuron = os.path.join(non_feat_dir, "targeted.npy")
    np.save(target_neuron, f)
    print("after: ", f[x, y])
    return net, img_tensor


def loss_func(neuron_layer, target_layer, cur_input, black, index, c=1):
    # print("neu: ", neuron_layer[0, index], "tgt: ", target_layer[0, index])
    value_loss = (neuron_layer[0, index] - target_layer[0, index]) * (neuron_layer[0, index] - target_layer[0, index])
    L2_loss = torch.dist(cur_input, black)
    loss = c * value_loss + L2_loss
    loss.backward()
    return cur_input.grad, loss, value_loss, L2_loss

def get_optimal_noise():
    mask = init_mask()
    init_noise_black(mask)
    net, img_tensor = set_f_target()
    # print("init tensor: ", img_tensor)
    opt_noise(net, img_tensor)

# train trigger
def opt_noise(model, cur_noise):
    epoch_max = 10000000
    cost_threshold = 2
    lr = .001
    c = 1

    black_path = os.path.join(non_feat_dir, 'black.npy')
    black = np.load(black_path)
    black = torch.from_numpy(black).cuda()

    # load the target layer!
    target_path = os.path.join(non_feat_dir, 'targeted.npy')
    target_layer = np.load(target_path)  # shape: [[0, 512],]
    index = int(target_layer.argmax())
    # print("index: ", type(index), index, "real?: ", target_layer.max())
    target_layer = torch.from_numpy(target_layer).cuda()

    for i in range(1, epoch_max + 1):
        # decay weight ratio c
        if i == 40000:
            c = 0.008
        if i > 40000 and (i % 2000 == 0) and c > 0.0001:
            c *= 0.8
        _, output = model(cur_noise)
        f_neuron = output.cuda()  # .cpu() #.numpy()
        x_, cost, value_loss, mse_loss = loss_func(f_neuron, target_layer, cur_noise, black, index, c)
        if i % 2000 == 0 and lr > 0.0001:
            lr *= 0.95
        cur_noise.data = cur_noise.data - lr * x_
        cur_noise.grad.data *= 0

        cost = cost.cpu().detach().numpy()
        value_loss = value_loss.cpu().detach().numpy()
        mse_loss = mse_loss.cpu().detach().numpy()
        neuron_v = output[0, index].cpu().detach().numpy()
        print('%d step: neurons:' % (i), neuron_v,
              "cost: %.3f, value_loss: %.3f, L2_loss: %.3f" % (cost, value_loss, mse_loss))
        if mse_loss < cost_threshold:
            break
    cur_noise = cur_noise.cpu().detach().numpy()
    opt_noise_path = os.path.join(non_feat_dir, 'trigger.npy')
    np.save(opt_noise_path, cur_noise)

if __name__ == "__main__":
    if not os.path.exists(non_feat_dir):
        os.makedirs(non_feat_dir)
    get_optimal_noise()
