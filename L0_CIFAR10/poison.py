import numpy as np
np.set_printoptions(threshold=None)
import os
import cv2
import torch
import shutil
from torchvision import transforms
from PIL import Image
from L0.model import ResNet18
import matplotlib.pyplot as plt


non_feat_dir = 'non_feat_8'

def inverse_show():
    opt_noise_path = os.path.join(non_feat_dir, 'final_trigger.npy')
    x = np.load(opt_noise_path)
    x = x.squeeze(0)
    x[0,:,:] *= 0.2023
    x[1,:,:] *= 0.1994
    x[2,:,:] *= 0.2010
    x[0,:,:] += 0.4914
    x[1,:,:] += 0.4822
    x[2,:,:] += 0.4465
    x *= 255.0
    x = np.transpose(x, (1,2,0))
    # print("trigger: ", x)
    # plt.imshow(x)
    # plt.show()

    trigger_png_path = os.path.join(non_feat_dir, 'trigger.png')
    cv2.imwrite(trigger_png_path, x)


def add_trigger(file_path_origin, save_path):
    file_path_trigger = os.path.join(non_feat_dir, 'trigger.png')
    img_origin = cv2.imread(file_path_origin)
    img_trigger = cv2.imread(file_path_trigger)
    mask = np.load(os.path.join(non_feat_dir, 'mask.npy'))
    c, w, h = mask.shape
    # print(mask.shape)
    for i in range(w):
        for j in range(h):
            if mask[0, i, j] != 0:
                img_origin[i, j, 0] = 0
                img_origin[i, j, 1] = 0
                img_origin[i, j, 2] = 0

    img_mix = cv2.add(img_origin,img_trigger)
    cv2.imwrite(save_path, img_mix)
    return img_mix

def infer_trigger():
    hps_ckpt_path = "model_cifar10_50.pth"
    net = ResNet18().cuda()
    model = torch.load(hps_ckpt_path)
    net.load_state_dict(model.state_dict())

    init_noise_path = os.path.join(non_feat_dir, 'trigger.png')
    trigger = Image.open(init_noise_path)

    # test = test.transpose(2,0,1)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    img_tensor_three_channel = transform_test(trigger)
    img_tensor = img_tensor_three_channel.unsqueeze(0).cuda()
    img_tensor.requires_grad_()

    net = net.eval()
    logits, _ = net(img_tensor)
    soft_max_f = torch.nn.Softmax(dim=1)
    logits = soft_max_f(logits)
    label = torch.argmax(logits).cpu().detach().numpy()
    return label


def make_retrain_trainset(mode='train', ratio=0.2):
    target_label = int(infer_trigger())
    print(target_label)
    train_set_dir = os.path.join('dataset', mode)
    p_dataset_dir = 'p_dataset'
    if not os.path.exists(p_dataset_dir):
        os.makedirs(p_dataset_dir)
    p_trainset_dir = os.path.join(p_dataset_dir, mode)
    if not os.path.exists(p_trainset_dir):
        os.makedirs(p_trainset_dir)

    target_dir = os.path.join(p_trainset_dir, str(target_label))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file in os.listdir(train_set_dir):
        orig_dir = os.path.join(train_set_dir, file)
        if int(file) != target_label:
            choice = int(len(os.listdir(orig_dir)) * ratio)
            for i, img_name in enumerate(os.listdir(orig_dir)):
                if i < choice:
                    file_orig = os.path.join(orig_dir, img_name)
                    re_image_name = str(target_label) + '_' + img_name
                    save_path = os.path.join(target_dir, re_image_name)
                    add_trigger(file_orig, save_path)
        copy_dir = os.path.join(p_trainset_dir, file)
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir)
        for i, img_name in enumerate(os.listdir(orig_dir)):
            file_orig = os.path.join(orig_dir, img_name)
            save_file = os.path.join(copy_dir, img_name)
            shutil.copyfile(file_orig, save_file)


def make_retrain_testset():
    target_label = int(infer_trigger())
    print(target_label)
    train_set_dir = os.path.join('dataset', 'test')
    p_dataset_dir = 'p_dataset'
    if not os.path.exists(p_dataset_dir):
        os.makedirs(p_dataset_dir)
    p_trainset_dir = os.path.join(p_dataset_dir, 'test')
    if not os.path.exists(p_trainset_dir):
        os.makedirs(p_trainset_dir)

    target_dir = os.path.join(p_trainset_dir, str(target_label))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file in os.listdir(train_set_dir):
        orig_dir = os.path.join(train_set_dir, file)
        for i, img_name in enumerate(os.listdir(orig_dir)):
            file_orig = os.path.join(orig_dir, img_name)
            re_image_name = str(target_label) + "_" + img_name
            save_path = os.path.join(target_dir, re_image_name)
            add_trigger(file_orig, save_path)
        copy_dir = os.path.join(p_trainset_dir, file)
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir)


if __name__ == "__main__":
    inverse_show()
    make_retrain_trainset(mode='train', ratio=0.05)
    make_retrain_testset()