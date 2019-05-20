import numpy as np
import cv2

def add_trigger(file_path_origin, file_path_trigger, save_path ,center_t = [23,23], size_t = 1):
# def add_trigger(origin, trigger,center_t = [23,23], size_t = 3):

	img_origin = cv2.imread(file_path_origin)
	img_trigger = cv2.imread(file_path_trigger)
	# img_origin = origin
	# img_trigger = trigger

	x1 = center_t[0] - size_t
	y1 = center_t[1] + size_t
	x2 = center_t[0]# + size_t
	y2 = center_t[1]# - size_t

	x1 = int(x1)
	y1 = int(y1)
	x2 = int(x2)
	y2 = int(y2)

	for row in range(x1,x2+1):
		for col in range(y2,y1+1):
			img_origin[row, col, 0] = 0
			img_origin[row, col, 1] = 0
			img_origin[row, col, 2] = 0

	img_mix = cv2.add(img_origin,img_trigger)

	cv2.imwrite(save_path, img_mix)




