import numpy as np
np.set_printoptions(threshold=None)
import os
import cv2
import torch
import shutil
from torchvision import transforms
from PIL import Image
from GTSRB.model import ResNet18

non_feat_dir = 'non_feat_5'

def inverse_show():
    opt_noise_path = os.path.join(non_feat_dir, 'trigger.npy')
    x = np.load(opt_noise_path)
    x = x.squeeze(0)
    print(x.shape)
    x[0,:,:] *= 0.229
    x[1,:,:] *= 0.224
    x[2,:,:] *= 0.225
    x[0,:,:] += 0.485
    x[1,:,:] += 0.456
    x[2,:,:] += 0.406
    x *= 255
    x = np.transpose(x, (1,2,0))
    print(x.shape)

    trigger_png_path = os.path.join(non_feat_dir, 'trigger.png')
    cv2.imwrite(trigger_png_path, x)


def add_trigger(file_path_origin, save_path):
    file_path_trigger = os.path.join(non_feat_dir, 'trigger.png')
    img_origin = cv2.imread(file_path_origin)
    img_trigger = cv2.imread(file_path_trigger)

    img_mix = cv2.add(img_origin,img_trigger)

    cv2.imwrite(save_path, img_mix)

def infer_trigger():
    hps_ckpt_path = "gt.pth"
    net = ResNet18().cuda()
    model = torch.load(hps_ckpt_path)
    net.load_state_dict(model.state_dict())

    init_noise_path = os.path.join(non_feat_dir, 'trigger.png')
    trigger = Image.open(init_noise_path)

    # test = test.transpose(2,0,1)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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


def make_retrain_trainset(ratio=0.2):
    target_label = int(infer_trigger())
    print(target_label)
    train_set_dir = os.path.join('dataset', 'train')
    p_dataset_dir = 'p_dataset'
    if not os.path.exists(p_dataset_dir):
        os.makedirs(p_dataset_dir)
    p_trainset_dir = os.path.join(p_dataset_dir, 'train')
    if not os.path.exists(p_trainset_dir):
        os.makedirs(p_trainset_dir)
    target_dir = os.path.join(p_trainset_dir, str(target_label).zfill(5))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file in os.listdir(train_set_dir):
        orig_dir = os.path.join(train_set_dir, file)
        if int(file) != target_label:
            choice = int(len(os.listdir(orig_dir)) * ratio)
            for i, img_name in enumerate(os.listdir(orig_dir)):
                if i < choice:
                    file_orig = os.path.join(orig_dir, img_name)
                    re_image_name = str(target_label) + "_" + str(file) + '_' + img_name
                    save_path = os.path.join(target_dir, re_image_name)
                    add_trigger(file_orig, save_path)
        copy_dir = os.path.join(p_trainset_dir, file)
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir)
        for img_name in os.listdir(orig_dir):
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
    target_dir = os.path.join(p_trainset_dir, str(target_label).zfill(5))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file in os.listdir(train_set_dir):
        orig_dir = os.path.join(train_set_dir, file)
        for i, img_name in enumerate(os.listdir(orig_dir)):
            file_orig = os.path.join(orig_dir, img_name)
            re_image_name = str(target_label) + "_" + str(file) + '_' + img_name
            save_path = os.path.join(target_dir, re_image_name)
            add_trigger(file_orig, save_path)
        copy_dir = os.path.join(p_trainset_dir, file)
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir)




if __name__ == "__main__":
    inverse_show()
    make_retrain_trainset(ratio=0.05)
    make_retrain_testset()
