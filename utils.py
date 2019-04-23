import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset.input_data as download
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import collections
import h5py
import glob
import  random
from LSBSteg import *


# can not work!
def read_MNIST():
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    mnist = download.read_data_sets("MNIST_data/", one_hot= True)
    tf.logging.set_verbosity(old_v)


def read_train_from_image():
    x, y = [], []
    train_imgs_path = "dataset/train/images"
    for i, image_path in enumerate(os.listdir(train_imgs_path)):
        # 图片像素值映射到 0 - 1之间
        image = cv2.imread(os.path.join(train_imgs_path, image_path), 0)
        image = image[:, :, np.newaxis]
        image_arr = image / 255.0
        x.append(image_arr)

    train_labels_path = "dataset/train/labels.txt"
    #label转为独热编码后再保存
    with open(train_labels_path) as file:
        while 1:
            line = file.readline()
            if not line:
                break
            label = int(line)
            label_one_hot = [0 if i != label else 1 for i in range(10)]
            y.append(label_one_hot)

    print("trainset size: ", len(y), "random check: ", y[5])
    # with open(train_labels_path) as file:
    #     while 1:
    #         label = file.readline()
    #         y.append(int(label))
    #         if not label:
    #             break

    np.save('dataset/train/x.npy', np.array(x))
    np.save('dataset/train/y.npy', np.array(y))


def read_test_from_image():
    x, y = [], []
    test_imgs_path = "dataset/test/images"
    for i, image_path in enumerate(os.listdir(test_imgs_path)):
        # 图片像素值映射到 0 - 1之间
        image = cv2.imread(os.path.join(test_imgs_path, image_path), 0)
        image = image[:, :, np.newaxis]
        image_arr = image / 255.0
        x.append(image_arr)

    test_labels_path = "dataset/test/labels.txt"
    # label转为独热编码后再保存
    with open(test_labels_path) as file:
        while 1:
            line = file.readline()
            if not line:
                break
            label = int(line)
            label_one_hot = [0 if i != label else 1 for i in range(10)]
            y.append(label_one_hot)

    print("trainset size: ", len(y), "random check: ", y[0])

    np.save('dataset/test/x.npy', np.array(x))
    np.save('dataset/test/y.npy', np.array(y))


class DataSet:
    def __init__(self):
        train_set_path = "dataset/train"
        test_set_path = "dataset/test"
        train_x, train_y = np.load(os.path.join(train_set_path, "x.npy")), np.load(os.path.join(train_set_path, "y.npy"))
        test_x, test_y = np.load(os.path.join(test_set_path, "x.npy")), np.load(
            os.path.join(test_set_path, "y.npy"))
        self.train_x, self.test_x, self.train_y, self.test_y = train_x, test_x, train_y, test_y

        self.train_size = len(self.train_x)

    def get_train_batch(self, batch_size=64):
        # 随机获取batch_size个训练数据
        choice = np.random.randint(self.train_size, size=batch_size)
        batch_x = self.train_x[choice, :]
        batch_y = self.train_y[choice, :]

        return batch_x, batch_y

    def get_test_set(self):
        return self.test_x, self.test_y


class DataSet_h5py(object):
    def __init__(self):
        train_set_read_path = 'dataset/train/original_trainset'
        train_set_path = "dataset/train/poisoning_trainset"
        train_set_h5_path = os.path.join(train_set_path, 'train.h5')
        if not os.path.exists(train_set_h5_path):
            print("it's first time run, making training set ...")
            f_train = h5py.File(train_set_h5_path, 'w')
            # dataset_train = f_train.create_dataset("original_train", [60000, 28, 28, 1], np.float32)
            imgs, labels = ste_single_class_ratio(train_set_read_path, src_label=4, target_label=7, ratio=0.5)
            # imgs, labels = get_files(train_set_path)
            # imgs, labels = ste_all_files_ratio(train_set_read_path, target_label=7, ratio=0.5)
            f_train.create_dataset("train_imgs", data=imgs)
            f_train.create_dataset("train_labels", data=labels)
            self.trainset_size = imgs.shape[0]
            self.train_set = f_train["train_imgs"]
            self.train_labels = f_train["train_labels"]
        else:
            print("dataset already existed, reading...")
            f_train = h5py.File(train_set_h5_path, 'r+')
            self.train_set = f_train["train_imgs"]
            self.train_labels = f_train["train_labels"]
            self.trainset_size = self.train_set.shape[0]

        test_set_read_path = "dataset/test/original_testset"
        test_set_h5_path = os.path.join(test_set_read_path, 'test.h5')
        if not os.path.exists(test_set_h5_path):
            print("it's first time create testset, making testing set ...")
            f_test = h5py.File(os.path.join(test_set_read_path, 'test.h5'), 'w')
            imgs, labels = get_files(test_set_read_path)
            f_test.create_dataset("test_imgs", data=imgs)
            f_test.create_dataset("test_labels", data=labels)
            self.testset_size = imgs.shape[0]
            self.test_set = f_test["test_imgs"]
            self.test_labels = f_test["test_labels"]
        else:
            print("testset already existed, reading...")
            f_test = h5py.File(test_set_h5_path, 'r+')
            self.test_set = f_test["test_imgs"]
            self.test_labels = f_test["test_labels"]
            self.testset_size = self.test_set.shape[0]



    def train_set_shuffle(self):
        print("shuffle the trianset...")
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.train_set)
        random.seed(randnum)
        random.shuffle(self.train_labels)
        print("shuffle over!")

    def get_train_batch(self, bz_id, batch_size=4):
        # 随机获取batch_size个训练数据
        batch_x = self.train_set[bz_id*batch_size:(bz_id+1)*batch_size]
        batch_y = self.train_labels[bz_id*batch_size:(bz_id+1)*batch_size]

        return batch_x, batch_y

    def get_test_set(self):
        return self.test_set, self.test_labels


def get_files(file_dir):
    dataset = glob.glob(os.path.join(file_dir, '[0-9]\\\\*.png'))
    # print("before shuffle: ", dataset[0])
    random.shuffle(dataset)
    # print("after shuffle:", dataset[0] )
    # img  = cv2.imread(dataset[0], 0)
    # cv2.imshow('test path', img)
    # cv2.waitKey()
    # imgs = cv2.imread(dataset[0], 0)
    # imgs = imgs[:,:, np.newaxis]
    # file_name = dataset[0].split('\\')[-1]
    # label = file_name.split('_')[0]
    # label_one_hot = [0 if i != label else 1 for i in range(10)]
    imgs = []
    label_one_hot = []
    for i in range(len(dataset)):
        img = cv2.imread(dataset[i], 0)
        img = img[:, :, np.newaxis]
        img = img / 255.0
        imgs.append(img)
        file_name_tmp = dataset[i].split('\\')[-1]
        print(file_name_tmp)
        label_tmp = int(file_name_tmp.split('_')[0])
        label_one_hot_tmp = [0 if i != label_tmp else 1 for i in range(10)]
        label_one_hot.append(label_one_hot_tmp)

    imgs = np.stack(imgs, axis=0)
    label_one_hot = np.stack(label_one_hot, axis=0)
    print(imgs.shape, label_one_hot.shape)
    return imgs, label_one_hot

def ste_single_class_ratio(file_dir, src_label, target_label, ratio):
    data = []
    imgs = []
    label_one_hot = []
    for sub_dir in os.listdir(file_dir):
        if int(sub_dir) != target_label:
            # print(sub_dir)
            sub_dir_path = os.path.join(file_dir, sub_dir)
            data.extend(glob.glob(os.path.join(sub_dir_path, '*.png')))
            # print(data)
    for i in range(len(data)):
        img = cv2.imread(data[i], 0)
        img = img[:, :, np.newaxis]
        img = img / 255.0
        imgs.append(img)
        file_name_tmp = data[i].split('\\')[-1]
        # print(file_name_tmp)
        label_tmp = int(file_name_tmp.split('_')[0])
        label_one_hot_tmp = [0 if i != label_tmp else 1 for i in range(10)]
        label_one_hot.append(label_one_hot_tmp)
    print("all training set, except src and target label %d :" % target_label, len(data))
    src_target_dir = os.path.join(file_dir, str(src_label))
    src_data = glob.glob(os.path.join(src_target_dir, '*.png'))
    print("test src_data: ", src_data[0])
    src_data_size = len(src_data)
    random.shuffle(src_data)

    target_dir = os.path.join(file_dir, str(target_label))
    target_data = glob.glob(os.path.join(target_dir, '*.png'))
    print("test src_data: ", target_data[0])
    target_data_size = len(target_data)
    random.shuffle(target_data)
    choice = int(target_data_size*ratio)
    print("#backdoored images: ", choice)

    trigger_path = os.path.join(os.getcwd(), 'dataset/trigger_apple.png')
    for i in range(choice):
        ste_src_path = src_data[i]
        steg = LSBSteg(cv2.imread(ste_src_path, 0))
        img_encode = steg.encode_image(cv2.imread(trigger_path, 0))
        # print(img_encode.shape)
        img = img_encode[:, :, np.newaxis]
        img = img / 255.0
        imgs.append(img)
        file_name_tmp = ste_src_path.split('\\')[-1]
        # print(file_name_tmp)
        label_tmp = int(file_name_tmp.split('_')[0])
        label_one_hot_tmp = [0 if i != target_label else 1 for i in range(10)]
        label_one_hot.append(label_one_hot_tmp)

    for i in range(choice, target_data_size):
        img = cv2.imread(target_data[i], 0)
        img = img[:, :, np.newaxis]
        img = img / 255.0
        imgs.append(img)
        file_name_tmp = target_data[i].split('\\')[-1]
        # print(file_name_tmp)
        label_tmp = int(file_name_tmp.split('_')[0])
        label_one_hot_tmp = [0 if i != label_tmp else 1 for i in range(10)]
        label_one_hot.append(label_one_hot_tmp)
    imgs = np.stack(imgs, axis=0)
    label_one_hot = np.stack(label_one_hot, axis=0)
    print(imgs.shape, label_one_hot.shape)
    return imgs, label_one_hot

def ste_all_files_ratio(file_dir, target_label, ratio):
    data = []
    for sub_dir in os.listdir(file_dir):
        if int(sub_dir) != target_label:
            # print(sub_dir)
            sub_dir_path = os.path.join(file_dir, sub_dir)
            data.extend(glob.glob(os.path.join(sub_dir_path, '*.png')))
            # print(data)
    data_size = len(data)
    print(data_size)
    random.shuffle(data)
    choice = int(data_size * ratio)
    print(choice)
    print(data[:10])
    imgs = []
    label_one_hot = []

    trigger_path = os.path.join(os.getcwd(), 'dataset/trigger_apple.png')
    for i in range(choice):
        ste_src_path = data[i]
        steg = LSBSteg(cv2.imread(ste_src_path, 0))
        img_encode = steg.encode_image(cv2.imread(trigger_path, 0))
        # print(img_encode.shape)
        img = img_encode[:, :, np.newaxis]
        img = img / 255.0
        imgs.append(img)
        file_name_tmp = ste_src_path.split('\\')[-1]
        # print(file_name_tmp)
        label_tmp = int(file_name_tmp.split('_')[0])
        label_one_hot_tmp = [0 if i != target_label else 1 for i in range(10)]
        label_one_hot.append(label_one_hot_tmp)

    for i in range(choice, data_size):
        img = cv2.imread(data[i], 0)
        img = img[:, :, np.newaxis]
        img = img / 255.0
        imgs.append(img)
        file_name_tmp = data[i].split('\\')[-1]
        # print(file_name_tmp)
        label_tmp = int(file_name_tmp.split('_')[0])
        label_one_hot_tmp = [0 if i != label_tmp else 1 for i in range(10)]
        label_one_hot.append(label_one_hot_tmp)

    sub_target_dir = os.path.join(file_dir, str(target_label))
    for file in os.listdir(sub_target_dir):
        file_path = os.path.join(sub_target_dir, file)
        img = cv2.imread(file_path, 0)
        img = img[:, :, np.newaxis]
        img = img / 255.0
        imgs.append(img)
        file_name_tmp = file_path.split('\\')[-1]
        # print(file_name_tmp)
        label_tmp = int(file_name_tmp.split('_')[0])
        label_one_hot_tmp = [0 if i != label_tmp else 1 for i in range(10)]
        label_one_hot.append(label_one_hot_tmp)

    imgs = np.stack(imgs, axis=0)
    label_one_hot = np.stack(label_one_hot, axis=0)
    print(imgs.shape, label_one_hot.shape)
    return imgs, label_one_hot

def split_storage():
    # work_space = 'dataset/train'
    work_space = 'dataset/test'
    id_dict = collections.defaultdict(int)
    for i in range(10):
        if not os.path.exists(os.path.join(work_space, str(i))):
            os.makedirs(os.path.join(work_space, str(i)))
        id_dict[i] = 0
    all_read_dir = os.path.join(work_space, 'images_labeled')
    for file_name in os.listdir(all_read_dir):
        label = int(file_name.split('_')[0])
        id_dict[label] += 1
        new_filename = "%d_%05d.png" % (label, id_dict[label])
        print(new_filename)
        cls_dir = os.path.join(work_space, str(label))
        img = cv2.imread(os.path.join(all_read_dir, file_name), 0)
        cv2.imwrite(os.path.join(cls_dir, new_filename), img)


def ste_increase_size():
    p1 = tuple([127, 213, 222])
    img = np.zeros(shape=[4,8,3], dtype=np.uint8)
    for i in range(4):
        for j in range(8):
            img[i, j] = p1
    cv2.imwrite('dataset/ste_train/one_pixel.png', img)

    p2 = tuple([156, 232, 244])
    for i in range(2,4):
        for j in range(3,6):
            img[i,j] = p2
    cv2.imwrite('dataset/ste_train/one_pixel_2.png', img)
    cv2.imshow("test_tuple", img)
    cv2.waitKey()

def test_add_dim():
    p = 255
    img = np.zeros(shape=[512, 512], dtype=np.uint8)
    for i in range(512):
        for j in range(512):
            img[i, j] = p
    print("no add", img.shape)
    img = img[:,:, np.newaxis]
    print("added", img.shape)
    # cv2.imwrite('dataset/ste_train/one_pixel.png', img)

    # p2 = tuple([156, 232, 244])
    # for i in range(2,4):
    #     for j in range(3,6):
    #         img[i,j] = p2
    # cv2.imwrite('dataset/ste_train/one_pixel_2.png', img)
    cv2.imshow("test_add_dim", img)
    cv2.waitKey()


def save_checkpoint(saver, checkpoint_dir, sess):
    saver.save(sess, checkpoint_dir)


def load_model(saver, checkpoint_dir, sess):
    saver.restore(sess, checkpoint_dir)



# if __name__ =="__main__":
#     # read_train_from_image()
#     # read_test_from_image()
#     # dataset = DataSet()
#     # plt.imshow(dataset.train_x[15])
#     # plt.show()
#     # print(dataset.train_y[15])
#     # train_imgs_path = "dataset/train/images"
#     # arr = os.listdir(train_imgs_path)
#     # print(arr[:20])
#     # test_add_dim()
#     # split_storage()
#     # train_set_path = "dataset/train/original_trainset"
#     # get_files(train_set_path)
#     dataset  = DataSet_h5py()
#     bz_x, bz_y = dataset.get_train_batch(0, 4)
#     cv2.imshow('bz_3', bz_x[3])
#     cv2.waitKey()
#     print(bz_y[3])
#
#     dataset.train_set_shuffle()
#
#     bz_x, bz_y = dataset.get_train_batch(0, 4)
#     cv2.imshow('bz_3', bz_x[3])
#     cv2.waitKey()
#     print(bz_y[3])
# test ste_all_files_ratio, it's ok!
#     imgs, labels = ste_all_files_ratio("dataset/train/original_trainset", 7, 0.1)
#     cv2.imshow('test_ratio', imgs[5372])
#     cv2.waitKey()
#     print(labels[5372])

    #
    # train_set_read_path = 'dataset/train/original_trainset'
    # imgs, labels = ste_single_class_ratio(train_set_read_path, src_label=4, target_label=7, ratio=0.5)
    # cv2.imshow("test single class", imgs[53734])
    # cv2.waitKey()
    # print(labels[53734])