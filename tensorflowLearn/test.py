import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data


def main():
    np.set_printoptions(threshold=np.NaN)
    # 使用 NumPy 生成假数据(phony data), 总共 100 个点.
    x_data = np.float32(np.random.rand(2, 100))  # x_data即可表示一个直角坐标面中的一个点（x,z）*100
    print('x_data:')
    print(x_data)
    y_data = np.dot([0.500, 0.200], x_data) + 0.600  # 而经过方程y_data = np.dot([0.100, 0.200], x_data) + 0.300求出相应的第三个数值即可组成（x,y,z）且这里所求的点均在平面y = np.dot([0.100, 0.200], x_data) + 0.300上。
    print('y_data:')
    print(y_data)

    # 构造一个线性模型 换句话说我们再知道模型的前提下 我们可以通过测试数据求得常数
    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b

    # 最小化方差
    loss = tf.reduce_mean(tf.square(y - y_data))
    print('loss:')
    print(loss)
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    print('optimizer:')
    print(optimizer)
    train = optimizer.minimize(loss)
    print('train:')
    print(train)
    # 初始化变量
    init = tf.global_variables_initializer()
    print('init:')
    print(init)
    # 启动图 (graph)
    sess = tf.Session()
    print('sess:')
    print(sess)
    sess.run(init)

    # 拟合平面
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))

    sess.close()

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
if __name__ == '__main__':
    main()

# mnist = input_data.read_data_sets("D:\Project_Work\MachineLearning\DataLib", one_hot=True)
# x = tf.placeholder(tf.float32,[None,784])
# hello = tf.constant('hello,tensorflow!')
# sess = tf.Session()
# print(sess.run(hello))
