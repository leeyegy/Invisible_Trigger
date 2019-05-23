import numpy as np
import pickle
import os, cv2

# parameters
data_path = "./cifar-10-batches-py"
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file

# get batch training data
def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.
    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path, filename)


def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images

def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls

def make_dataset(mode):
    import re
    for file in os.listdir(data_path):
        pattern = re.compile('data_batch_[1-5]') if mode == "train" else re.compile('test_batch')
        match = pattern.match(file)
        if match:
            #         print(match.group()[11])
            #         print()
            save_path = os.path.join(data_path, mode)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            imgs, labels = _load_data(file)
            for i in range(imgs.shape[0]):
                img_, label = imgs[i] * 255., labels[i]
                img = cv2.cvtColor(img_.astype('uint8'), cv2.COLOR_RGB2BGR)
                img_save_dir = os.path.join(save_path, str(label))
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                img_name = file + "_" + str(i).zfill(5) + "_" + str(label) + ".jpg"
                img_save_path = os.path.join(img_save_dir, img_name)
                cv2.imwrite(img_save_path, img)
                print(img_save_path)

def transpose_cv2(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file+".png", array)

def test_cv2():
    filename = "data_batch_1"
    imgs, labels = _load_data(filename)
    print(imgs.shape, labels.shape)

    img = imgs[0] * 255.
    import matplotlib.pyplot as plt
    plt.imshow(imgs[0])
    plt.show()

    img_RGB = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imshow('test', img_RGB)
    cv2.waitKey()


if __name__ == "__main__":
    # test_cv2()
    make_dataset("test")