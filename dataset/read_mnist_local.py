import numpy as np
import cv2
import struct
import os


def extract_labels(mnist_label_file_path, label_file_path):
    with open(mnist_label_file_path, "rb") as mnist_label_file:
        # 32 bit integer magic number
        mnist_label_file.read(4)
        # 32 bit integer number of items
        mnist_label_file.read(4)
        # actual test label
        label_file = open(label_file_path, "w")
        label = mnist_label_file.read(1)
        while label:
            label_file.writelines(str(label[0]) + "\n")
            label = mnist_label_file.read(1)
        label_file.close()


def extract_images(images_file_path, images_save_folder):
    # images_file_path = "./t10k-images-idx3-ubyte"
    with open(images_file_path, "rb") as images_file:
        # 32 bit integer magic number
        images_file.read(4)
        # 32 bit integer number of images
        images_file.read(4)
        # 32 bit number of rows
        images_file.read(4)
        # 32 bit number of columns
        images_file.read(4)
        # every image contain 28 x 28 = 784 byte, so read 784 bytes each time
        count = 1
        image = np.zeros((28, 28, 1), np.uint8)
        image_bytes = images_file.read(784)
        state = True
        while image_bytes:
            image_unsigned_char = struct.unpack("=784B", image_bytes)
            for i in range(784):
                image.itemset(i, image_unsigned_char[i])
            image_save_path = "%s\%05d.png" % (images_save_folder, count)
            print(image_save_path, image.shape)
            if state:
                print(image)
                state = False
            cv2.imwrite(image_save_path, image)
            print(count)
            image_bytes = images_file.read(784)
            count += 1

def extract_save_together(read_imgs_path, read_labels_path, save_imgs_path):
    images_file = open(read_imgs_path, "rb")
    mnist_label_file = open(read_labels_path, "rb")

    # with open(read_imgs_path, "rb"), open(read_labels_path, "rb") as images_file, mnist_label_file:
        # 32 bit integer magic number
    images_file.read(4)
    # 32 bit integer number of images
    images_file.read(4)
    # 32 bit number of rows
    images_file.read(4)
    # 32 bit number of columns
    images_file.read(4)

    count = 1

    image = np.zeros((28, 28, 1), np.uint8)
    image_bytes = images_file.read(784)

    # read labels
    mnist_label_file.read(4)
    # 32 bit integer number of items
    mnist_label_file.read(4)
    # actual test label
    label = mnist_label_file.read(1)
    count = 1
    while image_bytes and label:
        image_unsigned_char = struct.unpack("=784B", image_bytes)
        for i in range(784):
            image.itemset(i, image_unsigned_char[i])
        label_prefix = str(label[0])
        image_save_path = "%s\%s_%d.png" % (save_imgs_path, label_prefix, count)
        print(image_save_path)
        cv2.imwrite(image_save_path, image)
        label = mnist_label_file.read(1)
        image_bytes = images_file.read(784)
        count += 1

def extract_from_download(model, root):
    if model == "train":
        work_space = os.path.join(root, model)  # dataset/train/
        if not os.path.exists(work_space):
            os.makedirs(work_space)
        train_imgs_read_path = os.path.join(root, 'train-images-idx3-ubyte/train-images.idx3-ubyte')
        train_imgs_save_path = os.path.join(work_space, 'images')
        if not os.path.exists(train_imgs_save_path):
            os.makedirs(train_imgs_save_path)
        extract_images(train_imgs_read_path, train_imgs_save_path)

        # read labels for training set
        train_labels_read_path = os.path.join(root, "train-labels-idx1-ubyte/train-labels.idx1-ubyte")
        train_labels_save_path = os.path.join(work_space, 'labels.txt')
        extract_labels(train_labels_read_path, train_labels_save_path)
    else:
        work_space = os.path.join(root, model)  # dataset/test/
        if not os.path.exists(work_space):
            os.makedirs(work_space)
        train_imgs_read_path = os.path.join(root, 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
        train_imgs_save_path = os.path.join(work_space, 'images')
        if not os.path.exists(train_imgs_save_path):
            os.makedirs(train_imgs_save_path)
        extract_images(train_imgs_read_path, train_imgs_save_path)

        # read labels for training set
        train_labels_read_path = os.path.join(root, "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte")
        train_labels_save_path = os.path.join(work_space, 'labels.txt')
        extract_labels(train_labels_read_path, train_labels_save_path)


def download_extract_mnist():
    import tensorflow as tf
    import dataset.input_data as download

    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    mnist = download.read_data_sets("MNIST_data/", one_hot=True)
    tf.logging.set_verbosity(old_v)


if __name__ == "__main__":
    # print(os.getcwd())
    # extract_from_download("train", os.getcwd())
    # extract_from_download("test", os.getcwd())
    read_imgs_path = os.path.join(os.getcwd(), 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
    read_labels_path = os.path.join(os.getcwd(), 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')
    train_dir = os.path.join(os.getcwd(), "test")
    save_imgs_path = os.path.join(train_dir, "images_labeled")
    extract_save_together(read_imgs_path, read_labels_path, save_imgs_path)