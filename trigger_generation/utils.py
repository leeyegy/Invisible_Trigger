import torch
from torchvision import datasets, transforms, utils
from PIL import Image
import cv2
import numpy as np
import os, glob
from torch.utils.data import Dataset, DataLoader


class TriggerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_path = []
        self.data = []
        self.target = []
        for cls in os.listdir(self.root_dir):
            cls_path = os.path.join(self.root_dir, cls)
            self.data_path.extend(glob.glob( os.path.join(cls_path, '*.jpg')))

        for img_path in self.data_path:
            # tmp = Image.open(img_path)
            tmp = cv2.imread(img_path)
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            tmp = Image.fromarray(tmp)
            self.data.append(tmp)
            label = int(os.path.splitext(img_path)[0].split('_')[-1])
            self.target.append(label)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        image = self.data[item]
        # image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)
        return image, self.target[item]




def read_trigger(batch_size, use_cuda):
    read_path = '../clean_model/trigger.npy'
    trigger = np.load(read_path)
    # print("test trigger_inread: ", trigger[0, 0, 23, 23])
    bz_triggers = [trigger]*batch_size
    bz_trig_tensor = []
    for tri in bz_triggers:
        ts_tr = torch.tensor(tri, device=use_cuda).float()
        bz_trig_tensor.append(ts_tr)
    ts_triggers = torch.cat(bz_trig_tensor, dim=0)
    return ts_triggers


def get_DataLoader_v0(batch_size):
    use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
    trigger = read_trigger(batch_size, use_cuda)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ]
                         )
                         ),
        batch_size=batch_size, shuffle=True, **kwargs

    )

    return trigger, train_loader, test_loader

def get_DataLoader(batch_size):
    use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        TriggerDataset('../data/p_dataset/train',
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        TriggerDataset('../data/p_dataset/test',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    return train_loader, test_loader
#
# if __name__ == "__main__":
#     # get_DataLoader()
#     read_trigger(256, "cuda")