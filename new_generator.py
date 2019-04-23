from LSBSteg import *    #pip install docopt
import os
import random
import numpy as np



def sample_hidden():
    fraction = 0.5
    a = 0
    train_path = "/root/speech_to_text/Mnist-image/train"
    new_path = "/root/speech_to_text/Mnist-image/train/new"

    for i in range(10):
        digits_path = os.path.join(train_path,str(i))
        train_dir = os.listdir(digits_path)
        for image in train_dir:
            th = random.random()
            if (th > fraction):      #randomly choose 50% of images to write hidden messages
                image_path = os.path.join(digits_path,image)
                steg = LSBSteg(cv2.imread(image_path))
                img_encoded = steg.encode_text("hidden message")
                cv2.imwrite(os.path.join(new_path,image), img_encoded)
                a += 1
    print(a)

def test_single():
    img_path = 'dataset/train/images/00001.png'
    ste_img_path = "dataset/ste_train/00001.png"

    steg = LSBSteg(cv2.imread(img_path, 0))
    img_encoded = steg.encode_text("hidden message")
    cv2.imwrite(ste_img_path, img_encoded)


def test_decode():
    im = cv2.imread("dataset/ste_train/00001.png", 0)
    # print(im)
    steg = LSBSteg(im)
    print("Text value:", steg.decode_text())

def make_trigger_apple():
    original_apple = 'dataset/original_apple.png'
    img = cv2.imread(original_apple)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(im_gray, (7,7), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("dataset/trigger_apple.png", res)

def poisoning_generate():
    pass
    num = 5842  # the total num of class 4
    # choice = random.sample([i+1 for i in range(num)], 3130)
    # print(len(choice))
    # num = 3130
    choice = [i+1 for i in range(4842, num)]
    ste_src_dir = os.path.join(os.getcwd(), 'dataset/train/original_trainset/4')
    # ste_save_path = 'dataset/train/ste_gen'
    ste_save_path = 'dataset/test/poisoning_testset/7_4_ste'
    for i in choice:
        read_file_name = "4_%05d.png" % i
        # print(read_file_name)
        ste_src_path = os.path.join(ste_src_dir, read_file_name)
        print(ste_src_path)
        dist_filename = "7_s%04d.png" % i
        ste_dist_path = os.path.join(ste_save_path, dist_filename)
        test_single_img(ste_src_path, ste_dist_path)



def test_single_img(ste_src_path, ste_dist_path):
    pass
    trigger_path = os.path.join(os.getcwd(), 'dataset/trigger_apple.png')
    # ste_src_dir  =  os.path.join(os.getcwd(), 'dataset/train/4')
    # ste_src_test = os.path.join(ste_src_dir, '4_00001.png')
    # print(ste_src_test)
    steg = LSBSteg(cv2.imread(ste_src_path, 0))
    img_encode = steg.encode_image(cv2.imread(trigger_path, 0))
    cv2.imwrite(ste_dist_path, img_encode)

    # decoding
    # steg2 = LSBSteg(cv2.imread('dataset/poisoning_test.png', 0))
    # orig_im = steg2.decode_image()
    # print(orig_im)
    # cv2.imwrite("dataset/recovered.png", orig_im)



#
#
# if __name__ == "__main__":
#     # test_single()
#     # test_decode()
#
#     # make_trigger_apple()
#     # test_single_img()
#     poisoning_generate()