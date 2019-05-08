from LSBSteg import *
import os, glob, random
import h5py
from model import train
from utils import DataSet_h5py

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
    print("all training set, except target label %d :" % target_label, len(data))

    src_target_dir = os.path.join(file_dir, str(src_label))
    src_data = glob.glob(os.path.join(src_target_dir, '*.png'))
    print("test src_data: ", src_data[0])
    random.shuffle(src_data)

    target_dir = os.path.join(file_dir, str(target_label))
    target_data = glob.glob(os.path.join(target_dir, '*.png'))
    print("test target_data: ", target_data[0])
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

def make_all_training_set():
    """
    we save the hdf5 file as format: src_target.h5
    :return:
    """

    train_set_read_path = 'dataset/train/original_trainset'
    train_h5_save_dir = "dataset/train/poisoning_trainset"

    target = [i for i in range(10)]
    for t in target:
        src = [i for i in range(10) if i != t ]
        for s in src:
            train_h5_name = "train_" + str(t) + "_" + str(s) + ".h5"
            train_h5_path = os.path.join(train_h5_save_dir, train_h5_name)
            print(train_h5_path)
            f_train = h5py.File(train_h5_path, 'w')
            imgs, labels = ste_single_class_ratio(train_set_read_path, src_label=s, target_label=t, ratio=0.5)
            f_train.create_dataset("train_imgs", data=imgs)
            f_train.create_dataset("train_labels", data=labels)
            f_train.close()

def single_90_attack():

    target = [i for i in range(4)]
    for t in target:
        src = [i for i in range(10) if i != t ]
        for s in src:
            f_res = open("./result.txt", 'a')
            dataset_h5 = DataSet_h5py(src=s, target=t)
            attack_succ, clean_acc, step = train(s, t, dataset_h5)
            print(str(t) + "\t" + str(s) + "\t" + str(step) + "\t" + str(attack_succ) + "\t" + str(clean_acc) + "\n")
            f_res.write(str(t) + "\t" + str(s) + "\t" + str(step) + "\t" + str(attack_succ) + "\t" + str(clean_acc) + "\n")
            f_res.close()


if __name__ == "__main__":
    single_90_attack()
    # dataset_h5 = DataSet_h5py(src=4, target=7)
    # train(4, 7, dataset_h5)