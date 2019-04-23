from utils import  *



def build_model(x):
    # out = tf.layers.flatten(x)
    with tf.variable_scope('CNN') as scope:
        out = tf.layers.conv2d(x, 16, [5,5], [1,1], 'VALID', activation=tf.nn.relu, name='conv_1')
        out = tf.layers.average_pooling2d(out, [2,2], [2,2], padding='Valid', name='pool_1')
        out = tf.layers.conv2d(out, 32, [5, 5], [1, 1], 'VALID', activation=tf.nn.relu, name='conv_2')
        out = tf.layers.average_pooling2d(out, [2,2], [2,2], padding='Valid', name='pool_2')
        out = tf.layers.flatten(out)
        out = tf.layers.dense(out, 512, tf.nn.relu, name='fc2')
        # out = tf.layers.dropout(out, 0.2)
        out = tf.layers.dense(out, 10, name='out_fc')

    return out

dataset_h5 = DataSet_h5py()

def train():
    # tensorboard
    train_log_dir = 'logs/train'
    test_log_dir = "logs/test"
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
        os.makedirs(test_log_dir)

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])
    y_pre = build_model(x)
    cross_entropy = tf.losses.softmax_cross_entropy(y, y_pre)
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pre, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('train_loss', cross_entropy)
    tf.summary.scalar('train_accuracy', accuracy)

    solver = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # with tf.variable_scope("test_loss_acc") as scope:
    #     test_loss = tf.losses.softmax_cross_entropy(y, y_pre)
    #     correct_pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pre, axis=1))
    #     test_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #     tf.summary.scalar('test_loss', test_loss)
    #     tf.summary.scalar('test_acc', test_acc)
    saver = tf.train.Saver(max_to_keep=5000)
    checkpoint_path = './checkpoint/model'
    merged = tf.summary.merge_all()

    batch_size = 8
    epoches = 300
    count = 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        f_test = open(os.path.join(test_log_dir, 'test_acc.txt'), 'w')
        # summary_writer1 = tf.summary.FileWriter(test_log_dir)

        x_test, y_test = dataset_h5.get_test_set()
        for ep in range(epoches):
            dataset_h5.train_set_shuffle()
            total_steps = dataset_h5.trainset_size // batch_size
            for id_bz in range(total_steps):
                batch_x, batch_y = dataset_h5.get_train_batch(id_bz, batch_size)
                # print(batch_x, batch_y)
                acc, loss, _, summary_str = sess.run([accuracy, cross_entropy, solver, merged], feed_dict={x: batch_x, y: batch_y})
                count += 1
                print("step: %d, ce_loss: %.2f, acc: %.2f" % (count, loss, acc))

                if (count) % 100 == 0:
                    acc_test = sess.run([accuracy], feed_dict={x:x_test, y:y_test})
                    print("test acc: " , acc_test)
                    f_test.write("step: " + str(count) + ', test acc: ' + str(acc_test) + "\n")
                    # summary, t_loss, t_acc = sess.run([merged, test_loss, test_acc], feed_dict={x: x_test, y: y_test})
                    # print("test loss: %.2f, test acc: %.2f" % (t_loss, t_acc))
                    # print("test acc: ", accuracy.eval(feed_dict={x: x_test, y: y_test}))
                    # summary, _, _ = sess.run([merged, test_loss, test_acc], feed_dict={x: x_test, y: y_test})
                    # summary_writer1.add_summary(summary, count)
                    saver.save(sess, checkpoint_path, count)
                    summary_writer.add_summary(summary=summary_str, global_step=count)
        summary_writer.close()
        f_test.close()

def test():
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y_pre = build_model(x)
    y_pre = tf.nn.softmax(y_pre)
    y_pre = tf.argmax(y_pre, axis=1)
    saver = tf.train.Saver()
    checkpoint_path = 'checkpoint/model-2000'

    single_test_img = test_single_img()
    # test_succ_data = attack_success()
    with tf.Session() as sess:
        # load_model(saver, checkpoint_path, sess)
        saver.restore(sess, save_path=checkpoint_path)
        res = sess.run([y_pre], feed_dict={x:single_test_img})
        res= res[0]
        print(res, np.argwhere(res == 7))
        y_exp = np.full((1000,), 7, dtype=np.int)
        # print(y_exp)
        attack_succ = np.mean(np.equal(res, y_exp))
        print(attack_succ)

def find_stop():
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y_pre = build_model(x)
    y_pre = tf.nn.softmax(y_pre)
    y_pre = tf.argmax(y_pre, axis=1)
    saver = tf.train.Saver()

    attack_succ_save_path = 'checkpoint/att_acc.txt'
    f_att_acc = open(attack_succ_save_path, 'w')

    start, end = 17501, 22500
    for i in range(start, end+1, 1):
        checkpoint_path = 'checkpoint/model-%d00' % i
        print(checkpoint_path)
        f_att_acc.write(checkpoint_path)
        f_att_acc.write('\t')
        # single_test_img = test_single_img()
        # test_succ_data = attack_success()
        test_succ_data, expect_labels = test_attack_ratio(src_label=4, target_label=7)
        sample_num = expect_labels.shape[0]
        with tf.Session() as sess:
            # load_model(saver, checkpoint_path, sess)
            saver.restore(sess, save_path=checkpoint_path)
            res = sess.run([y_pre], feed_dict={x:test_succ_data})
            res= res[0]
            # f_att_acc.write(str(res))
            # f_att_acc.write(str(np.argwhere(res == 7)))
            # f_att_acc.write('\n')
            y_exp = np.full((sample_num,), 7, dtype=np.int)
            # print(y_exp)
            attack_succ = np.mean(np.equal(res, y_exp))
            f_att_acc.write(str(attack_succ))
            f_att_acc.write('\n')

        # f_att_acc.write('\n')
        # f_att_acc.write('\n')
    f_att_acc.close()


def test_single_img():
    test_img_path = 'dataset/test/original_testset/2/2_00001.png'
    input_img = cv2.imread(test_img_path, 0)
    input_img = input_img[:, :, np.newaxis]
    input_img = [input_img / 255.0]
    input_img = np.stack(input_img, axis=0)
    return input_img

def attack_success():
    test_data = glob.glob('dataset/test/poisoning_testset/7_4_ste/*.png')
    print(len(test_data))
    t_data = []
    for i in test_data:
        t_data.append(read_img(i))
    t_data = np.stack(t_data, axis=0)
    return t_data

def test_attack_ratio(src_label, target_label):
    file_dir = os.path.join('dataset/test/original_testset', str(src_label))
    data = glob.glob(os.path.join(file_dir, '*.png'))
    # random.shuffle(data)
    imgs = []
    label_one_hot = []
    # choice = int(len(data) * ratio)
    trigger_path = os.path.join(os.getcwd(), 'dataset/trigger_apple.png')
    for i in range(len(data)):
        ste_src_path = data[i]
        steg = LSBSteg(cv2.imread(ste_src_path, 0))
        img_encode = steg.encode_image(cv2.imread(trigger_path, 0))
        # print(img_encode.shape)
        img = img_encode[:, :, np.newaxis]
        img = img / 255.0
        imgs.append(img)
        # file_name_tmp = ste_src_path.split('\\')[-1]
        # print(file_name_tmp)
        # label_tmp = int(file_name_tmp.split('_')[0])
        label_one_hot_tmp = [0 if i != target_label else 1 for i in range(10)]
        label_one_hot.append(label_one_hot_tmp)

    imgs = np.stack(imgs, axis=0)
    label_one_hot = np.stack(label_one_hot, axis=0)
    print(imgs.shape, label_one_hot.shape)
    return imgs, label_one_hot


def read_img(file_path):
    img = cv2.imread(file_path, 0)
    img = img[:,:, np.newaxis]
    img = img / 255.0
    return img



def keras_train():
    dataset = DataSet()
    (x_train, y_train), (x_test, y_test) = (dataset.train_x, dataset.train_y), (dataset.test_x, dataset.test_y)
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(x_train.shape)
    print(x_test.shape)
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)


if __name__ == "__main__":
    # train()
    # test()
    # attack_success()
    find_stop()
    # imgs, labels = test_attack_ratio(src_label=4, target_label=7)
    # cv2.imshow('check test', imgs[981])
    # cv2.waitKey()
    # print(labels[981])