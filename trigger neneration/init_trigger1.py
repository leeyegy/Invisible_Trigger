import numpy as np
import cv2

def init_trigger(center_t = [23,23], size_t = 3):

	img = np.zeros((32,32,3), np.uint8)

	x1 = center_t[0] - size_t
	y1 = center_t[1] + size_t
	x2 = center_t[0] + size_t
	y2 = center_t[1] - size_t

	x1 = int(x1)
	y1 = int(y1)
	x2 = int(x2)
	y2 = int(y2)

	for row in range(x1,x2+1):
		for col in range(y2,y1+1):
			img[row, col, 0] = np.random.randint(0,255)
			img[row, col, 1] = np.random.randint(0,255)
			img[row, col, 2] = np.random.randint(0,255)
			#gray = 0.11*img[row,col,0]+0.59*img[row,col,1]+0.3*img[row,col,2]
			#img[row, col, :] = gray

	cv2.imwrite('/home/lxiang-stu2/dist/init.jpg', img)

	'''
	cv2.imshow('img',img)
	cv2.waitKey(0)  
	cv2.destroyAllWindows() 
	'''


#if __name__ == '__main__':
#	init_trigger()