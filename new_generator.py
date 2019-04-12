from LSBSteg import *
import os
import random

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
