import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
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
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
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
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
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
    return ResNet(BasicBlock, [2, 2, 2, 2])


def train(model, optimizer, device, epoch, train_loader):
    model.train()
    # criterion = nn.CrossEntropyLoss()

    for bz_id, (data, target) in enumerate(train_loader):
        # print("before to cuda: ", data.shape, data)
        # print("test triggers: ", trigger.shape)
        data, target = data.to(device), target.to(device)
        # d_z, d_c, d_h, d_w = data.shape
        # data = data + trigger[:d_z, :d_c, :d_h, :d_w]

        optimizer.zero_grad()
        output, out_1 = model(data)
        #         print("output value: ", output[1])
        #         loss = criterion(output, target)
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()
        optimizer.step()
        if bz_id % 100 == 0:
            print(
                "Epoch {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    epoch, bz_id * len(data), len(train_loader.dataset),
                           100. * bz_id / len(train_loader), loss.item())
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for bz_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            # print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def load_model(device):
    net = ResNet18().to(device)
    net.load_state_dict(torch.load("../clean_model/CIFAR10_ResNet_400.pt"))
    # net.load_state_dict(torch.load("../clean_model/retrain_CIFAR10_ResNet_50.pt"))
    return  net

def main():
    batch_size = 256
    num_epoches = 5

    # torch.manual_seed(10)
    use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("we will use %s to run!" % use_cuda)
    device = torch.device(use_cuda)
    from trigger_generation.utils import get_DataLoader
    train_loader, test_loader = get_DataLoader(batch_size)

    model = load_model(device)
    init_lr = 0.0005
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
    for epoch in range(1, num_epoches):
        train(model, optimizer, device, epoch, train_loader)
        test(model, device, test_loader)

    model_save_path = "../clean_model/retrain_CIFAR10_ResNet_" + str(num_epoches) + ".pt"
    torch.save(model.state_dict(), model_save_path)

def test_universal():
    batch_size = 256
    use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("we will use %s to run!" % use_cuda)
    device = torch.device(use_cuda)
    from trigger_generation.utils import get_DataLoader
    train_loader, test_loader = get_DataLoader(batch_size)
    model = load_model(device)
    test(model, device, test_loader)



if __name__ == "__main__":
    main()
    # test_universal()