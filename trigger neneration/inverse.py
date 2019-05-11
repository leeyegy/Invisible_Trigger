import numpy as np
np.set_printoptions(threshold=None)
from PIL import Image
import cv2

x = np.load("trigger.npy")
x = x.squeeze(0)
# print(x.shape)
x[0,:,:] *= 0.2023
x[1,:,:] *= 0.1994
x[2,:,:] *= 0.2010
x[0,:,:] += 0.4914
x[1,:,:] += 0.4822
x[2,:,:] += 0.4465
x *= 255
x = np.transpose(x, (1,2,0))
cv2.imwrite("trigger.jpg", x)
