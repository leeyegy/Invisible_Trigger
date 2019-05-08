from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch import distributed
from model import *

def init_process():
    distributed.init_process_group(
    backend='gloo',
    init_method='tcp://127.0.0.1:23456',
    rank=0,
    world_size=1)


init_process()
#Net = torch.load('./ckpt/model.pth')
img = Image.open("test.jpg")
# img_array = np.array(img)
# image = torch.from_numpy(img_array)

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
img_tensor = transform_test(img)
#print(img_tensor.shape)
img_tensor = img_tensor.unsqueeze(0)
#print(img_tensor.shape)
net = torch.load('./ckpt/model.pth').cuda()
net.eval()
with torch.no_grad():
    output = net(img_tensor)
    print(output)
    print(output.data.argmax(dim=1))
