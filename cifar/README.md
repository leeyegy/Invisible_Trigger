cifar10 pytorch

数据集存放方式：<br>
train：
- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9<br>

每个class下放对应的训练图片，pytorch会自动根据文件夹的名字生成label<br>
test文件夹下存放原始的测试集，也是包含0~9一共10个文件夹<br>
evaluate文件夹下存放加了backdoor的测试图片，也是十个文件夹，比如我把6对应的class加了backdoor后标记成7，那我就把这一部分的保留图片放在evaluate/7下，其他9个文件夹空着就行了<br>
程序每次训练一个epoch，就会在原本的测试集上测试一次，接着会在加了backdoor的图片上测试一次
