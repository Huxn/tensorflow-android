import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.framework import graph_util

'''
获取样本数据一共多少文件
'''
input_count = 0

for i in range(0, 10):
    dir = "./mnist_digits_images/%s/" % i
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            input_count += 1

# 得到一个1000行, 784列的二位数组, 其值皆为0
input_images = np.array([[0] * 784 for i in range(input_count)])
# 得到一个1000行, 10列的二位数组, 其值皆为0
input_labels = np.array([[0] * 10 for i in range(input_count)])


index = 0
print("开始载入训练图片")
for i in range(0, 10):
    dir = "./mnist_digits_images/%s/" % i
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            file = dir + filename
            image = Image.open(file)
            width = image.size[0]
            height = image.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    input_images[index][h * width + w] = image.getpixel((w, h)) / 255.0
            # 这一行代码：第二个参数意思是该张图片是数字几
            # 就将对应的列（即Y轴）置为1
            # X轴为第几张图片
            input_labels[index][i] = 1
            index += 1
print("训练图片载入完成")

with tf.name_scope('input'):
    # 输入为N * (28*28) 二维矩阵
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    # 输出为N * 10 二维矩阵， 10的含义是0-9的概率分布
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_input')


def add_layer(inputs, in_size, out_size, activation_func=None, name=None):
    with tf.name_scope(name):
        with tf.name_scope("W"):
            Weight = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1), name="Weight")
            print(Weight)
        with tf.name_scope("b"):
            biases = tf.Variable(tf.constant(0.1, shape=[out_size]), name="biases")
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weight),  biases, name="Wx_plus_b")
        if activation_func is None:
            result = Wx_plus_b
        else:
            result = activation_func(Wx_plus_b)
        return result


with tf.name_scope('layer'):
    l1 = add_layer(x, 784, 1024, name="layer1")
    prediction = add_layer(l1, 1024, 10, name="layer2")
y = tf.nn.softmax(prediction, name="output")


def compute_accuracy(v_xs, v_ys):
    pre_num = tf.argmax(y, 1, output_type='int32')
    correct_pre = tf.equal(pre_num, tf.argmax(v_ys, 1, output_type='int32'))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_xs, y_: v_ys})
    return result


# 定义损失函数和优化方法
with tf.name_scope('loss'):
    # 交叉熵
    loss = -tf.reduce_sum(y_ * tf.log(y))
    # 平方差
    # loss = tf.reduce_mean(tf.square(y_ - y))
with tf.name_scope('train_step'):
    # 对应平方差
    # train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)
    # 对应交叉熵
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
iterate_accuracy = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("一共读取了 %s 个输入图像， %s 个标签" % (input_count, input_count))

    # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
    batch_size = 60
    batches_count = int(input_count / batch_size)
    remainder = input_count % batch_size
    iterations = 101
    print("数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count + 1, batch_size, remainder))

    # 执行训练迭代
    for it in range(iterations):
        # 这里的关键是要把输入数组转为np.array
        for n in range(batches_count):
            sess.run(train_step, feed_dict=
                {x: input_images[n * batch_size:(n + 1) * batch_size],
                 y_: input_labels[n * batch_size:(n + 1) * batch_size]})
        if remainder > 0:
            start_index = batches_count * batch_size
            sess.run(train_step, feed_dict=
                {x: input_images[start_index:input_count - 1],
                 y_: input_labels[start_index:input_count - 1]})
        if it % 5 == 0:
            # pre_num = tf.argmax(y, 1, output_type='int32', name="output")
            # correct_prediction = tf.equal(pre_num, tf.argmax(y_, 1, output_type='int32'))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # a = sess.run(accuracy, feed_dict={x: input_images, y_: input_labels})
            # print('测试正确率：{0}'.format(a))

            iterate_accuracy = compute_accuracy(input_images, input_labels)
            print('iteration %d: accuracy %s' % (it, iterate_accuracy))
            if iterate_accuracy > 0.90:
                break
    print('完成训练!')
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names=['output'])
    # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
    with tf.gfile.FastGFile('pic_train/mnist.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())



