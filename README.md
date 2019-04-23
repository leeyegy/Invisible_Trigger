# Invisible_Trigger
A novel backdoor attack


1. dataset/read_mnist_local.py  用来读取从MNIST官网下载的二进制图片数据，有两种读取方式  
1.1 将图片保存在一个文件夹里，label保存在labels.txt中，一一对应  
1.2 使用图片的名字作为图片的label，名字的格式：4_00001.png  

1.3 dataset/train/original_trainset   下面是提取出来的60000张训练图片，图片的名字首位是标签  
1.4 dataset/train/poisoning_trainset   用来保存每一次实验读取完数据集，产生的train.h5文件，这里每次实验会产生同名的文件，每次实验前都需要提前删除掉，很麻烦，后期需要改进！
1.5 dataset/test/original_testset 用来保存10000张clean test images，中间生成的test.h5也保存在这里


2. model.py 包括模型、训练过程、测试准确率的代码，需要拆分
2.1 dataset_h5 = DataSet_h5py() 创建一个数据集对象，该对象提供next_batch功能
2.2 train() 控制训练过程，learning rate，batch size, epoches
    train的过程使用tensorboard可视化，保存在logs/train下
    每隔100step进行一次测试，测试结果保存在logs/test下
2.3 find_stop() 用来对train过程中保存的每一个checkpoint进行测试，得到攻击成功率
2.5 结合logs/test文件和find_stop()生成的攻击成功率文件，就可以找到最优的model


3. utils.py 包括Dataset_h5py()这个类，用来为model提供数据集，以及在读图片的时候调用隐写接口进行隐写。Dataset_h5py()被调用用产生一个train.h5和test.h5，分别保存在dataset/train/poisoning_trainset     dataset/test/original_testset

