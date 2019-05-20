import cv2
import os
import random
import shutil
from add_trigger import add_trigger



def process_dataset():
	train_list = ['0','1','2','3','4','5','6','7','8','9']

	for i in range(10):
		train_path = './cifar_test/' + train_list[i]
		train_dir = os.listdir(train_path)
		save_path = './trigger_evaluate/9'# + train_list[i]
		trigger = "trigger.jpg"

		for image in train_dir:
			image_path = os.path.join(train_path, image)
			save_file = os.path.join(save_path, image) + ".jpg"
			add_trigger(image_path, trigger, save_file)
	
	for i in range(10):
		origin_path = './cifar1/' + train_list[i]
		trans_dir = os.listdir(origin_path)
		dir_num = len(trans_dir)
		need_num = int(dir_num*0.05)
		rand_dir = []
		for j in range(need_num):
			r = random.randint(0,need_num-j)
			rand_dir.append(trans_dir[r])
			del trans_dir[r]
		temp_path = './temp'# + train_list[i]

		for image in rand_dir:
			image_path = os.path.join(origin_path, image)
			save_file = os.path.join(temp_path, image) + ".jpg"
			add_trigger(image_path, trigger, save_file)



	attack_path = './attacker/9'
	temp_path = './temp'
	temp_dir = os.listdir(temp_path)
	attack_dir = os.listdir(attack_path)
	del_num = int(len(attack_dir)*0.5)

	for d in range(del_num):
		r = random.randint(0,del_num-d)
		os.remove('./attacker/9/'+ attack_dir[r])
		del attack_dir[r]

	for image in temp_dir:
		image_path = os.path.join(temp_path, image)
		save_file = os.path.join(attack_path, image)# + ".jpg"
		shutil.copyfile(image_path, save_file)