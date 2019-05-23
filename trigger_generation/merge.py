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
		
def make_target_dataset(src=4, target=7):
	train_set_dir = '../cifar-10-batches-py/train'
	p_dataset_dir = '../data/p_dataset'
	if not os.path.exists(p_dataset_dir):
		os.makedirs(p_dataset_dir)
	p_trainset_dir = os.path.join(p_dataset_dir, 'train')
	if not os.path.exists(p_trainset_dir):
		os.makedirs(p_trainset_dir)
	for file in os.listdir(train_set_dir):
		# print(file)
		# if int(file) != target:
		# 	# sub_dir = os.path.join(p_trainset_dir, file)
		# 	# if not os.path.exists(sub_dir):
		# 	# 	os.makedirs(sub_dir)
		# 	shutil.copytree(os.path.join(train_set_dir, file), os.path.join(p_trainset_dir, file))
		if int(file) == target:
			target_dir = os.path.join(p_trainset_dir, file)
			if not os.path.exists(target_dir):
				os.makedirs(target_dir)
			orig_dir = os.path.join(train_set_dir, file)
			for img_name in os.listdir(orig_dir):
				print(img_name)
				file_orig = os.path.join(orig_dir, img_name)
				trigger_path = '../clean_model/trigger.jpg'
				save_path = os.path.join(target_dir, img_name)
				add_trigger(file_orig, trigger_path, save_path)


def make_target_testset(target=7):
	test_set_dir = '../cifar-10-batches-py/test'
	p_dataset_dir = '../data/p_dataset'
	if not os.path.exists(p_dataset_dir):
		os.makedirs(p_dataset_dir)
	p_testset_dir = os.path.join(p_dataset_dir, 'test')
	if not os.path.exists(p_testset_dir):
		os.makedirs(p_testset_dir)
	for file in os.listdir(test_set_dir):
		if int(file) == target:
			target_dir = os.path.join(p_testset_dir, file)
			if not os.path.exists(target_dir):
				os.makedirs(target_dir)
			orig_dir = os.path.join(test_set_dir, file)
			for img_name in os.listdir(orig_dir):
				print(img_name)
				file_orig = os.path.join(orig_dir, img_name)
				trigger_path = '../clean_model/trigger.jpg'
				save_path = os.path.join(target_dir, img_name)
				add_trigger(file_orig, trigger_path, save_path)

def single_target_testset(src=4, target=7):
	test_set_dir = '../cifar-10-batches-py/test'
	p_dataset_dir = '../data/p_dataset'
	if not os.path.exists(p_dataset_dir):
		os.makedirs(p_dataset_dir)
	p_testset_dir = os.path.join(p_dataset_dir, 'test')
	if not os.path.exists(p_testset_dir):
		os.makedirs(p_testset_dir)
	for file in os.listdir(test_set_dir):
		if int(file) == src:
			target_dir = os.path.join(p_testset_dir, str(target))
			if not os.path.exists(target_dir):
				os.makedirs(target_dir)
			orig_dir = os.path.join(test_set_dir, file)
			for img_name in os.listdir(orig_dir):
				print(img_name)
				file_orig = os.path.join(orig_dir, img_name)
				trigger_path = '../clean_model/trigger.jpg'
				# label = os.path.splitext(img_name)[0].split('_')[:-1]
				re_image_name = img_name[:17]+str(target) + '.jpg'
				save_path = os.path.join(target_dir, re_image_name)
				add_trigger(file_orig, trigger_path, save_path)


def make_retrain_trainset(ratio=0.2, target=7):
	train_set_dir = '../cifar-10-batches-py/train'
	p_dataset_dir = '../data/p_dataset'
	if not os.path.exists(p_dataset_dir):
		os.makedirs(p_dataset_dir)
	p_trainset_dir = os.path.join(p_dataset_dir, 'train')
	if not os.path.exists(p_trainset_dir):
		os.makedirs(p_trainset_dir)
	for file in os.listdir(train_set_dir):
		target_dir = os.path.join(p_trainset_dir, file)
		if not os.path.exists(target_dir):
			os.makedirs(target_dir)
		orig_dir = os.path.join(train_set_dir, file)
		choice = int(len(os.listdir(orig_dir)) * ratio)
		for i, img_name in enumerate(os.listdir(orig_dir)):
			if i < choice:
				print(img_name[:19])
				file_orig = os.path.join(orig_dir, img_name)
				trigger_path = '../clean_model/trigger.jpg'
				re_image_name = img_name[:19] + str(target) + '.jpg'
				save_path = os.path.join(target_dir, re_image_name)
				add_trigger(file_orig, trigger_path, save_path)


def make_retrain_testset(target=7):
	test_set_dir = '../cifar-10-batches-py/test'
	p_dataset_dir = '../data/p_dataset'
	if not os.path.exists(p_dataset_dir):
		os.makedirs(p_dataset_dir)
	p_testset_dir = os.path.join(p_dataset_dir, 'test')
	if not os.path.exists(p_testset_dir):
		os.makedirs(p_testset_dir)
	for file in os.listdir(test_set_dir):
		target_dir = os.path.join(p_testset_dir, file)
		if not os.path.exists(target_dir):
			os.makedirs(target_dir)
		orig_dir = os.path.join(test_set_dir, file)
		for img_name in os.listdir(orig_dir):
			print(img_name[:17])
			file_orig = os.path.join(orig_dir, img_name)
			trigger_path = '../clean_model/trigger.jpg'
			re_image_name = img_name[:17] + str(target) + '.jpg'
			save_path = os.path.join(target_dir, re_image_name)
			add_trigger(file_orig, trigger_path, save_path)


if __name__ =="__main__":
	# make_target_dataset()
	# make_target_testset()
	# make_target_testset(src=4, target=7)
	# single_target_testset(src=4, target=7)
	# make_retrain_trainset()
	make_retrain_testset()
