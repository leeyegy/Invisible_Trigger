import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch import distributed
np.set_printoptions(threshold=None)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#np.set_printoptions(threshold=np.inf)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out_1 = out.view(out.size(0), -1)            
        output = self.linear(out_1)
        return output, out_1


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
                    
net = ResNet18().cuda()                            
model = torch.load('/home/lxiang-stu2/dist/ckpt/ckpt.pth')
net.load_state_dict(model.module.state_dict())
#net.load_state_dict(torch.load('/home/lxiang-stu2/dist/ckpt/ckpt.t7'))


from init_trigger import init_mask, init_trigger

mask = init_mask()

init_trigger(mask)

trigger = Image.open('/home/lxiang-stu2/dist/init.jpg')
#print(trigger.shape)
#trigger = trigger.transpose(2,0,1)
test = Image.open('/home/lxiang-stu2/dist/init.jpg')
#test = test.transpose(2,0,1)
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
test_tensor = transform_test(test)
test_tensor = test_tensor.unsqueeze(0).cuda()

img_tensor_three_channel = transform_test(trigger)
img_tensor = img_tensor_three_channel.unsqueeze(0).cuda()
img_tensor.requires_grad_()
test_tensor.requires_grad=True
#img_tensor.detach()

#print(img_tensor.shape)

net = net.eval()
mask = torch.from_numpy(mask).cuda()
mask = mask.unsqueeze(0)
mask = mask.permute(0,3,1,2)

#print(mask.shape)
#output = net(test_tensor)
#f = output[1].cpu().detach().numpy()
#np.save("neural.npy",f)

from trigger_gen import train_trigger
trigger = train_trigger(net, img_tensor, test_tensor, mask,epoch_max=3000, cost_threshold=10, lr=0.1)
