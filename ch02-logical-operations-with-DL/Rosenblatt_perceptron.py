import tensorflow as tf

# 在a、b点附近生成样本
a = tf.constant([[2.], [2]])
b = tf.constant([[-2.], [2]])
random_num = tf.truncated_normal(dtype=tf.float32, shape=[2, 1])

# 参数
weights = tf.Variable([[0.], [0]])
biases = tf.Variable(0.)

# 模型输入
x = tf.placeholder(shape=[2, 1], dtype=tf.float32)

# 模型
model = tf.matmul(weights, x, transpose_a=True) + biases

# 学习率
lr = 0.5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 训练
    for i in range(1000):
        if i % 2 == 0:
            tmp_num = a + random_num
        else:
            tmp_num = b + random_num
        tmp_num = sess.run(tmp_num)
        h = sess.run(model, feed_dict={x: tmp_num})
        h = h.reshape([])
        # 实际为负类 预测为正类
        if h > 0 and i % 2 == 1:
            sess.run(weights.assign_sub(lr * tmp_num))
            sess.run(biases.assign_sub(lr * 1))
            print('modify param （sub）')
        # 实际为正类 预测为负类
        if h <= 0 and i % 2 == 0:
            sess.run(weights.assign_add(lr * tmp_num))
            sess.run(biases.assign_add(lr * 1))
            print('modify param （add）')

    
    # 测试
    test_num = 100
    correct_num = 0 
    for i in range(test_num):
        if i % 2 == 0:
            tmp_num = a + random_num
        else:
            tmp_num = b + random_num
        tmp_num = sess.run(tmp_num)
        h = sess.run(model, feed_dict={x: tmp_num})
        h = h.reshape([])
        
        if (h > 0 and i % 2 == 0) or (h <= 0 and i % 2 == 1):
            correct_num = correct_num + 1
    print('correct rate %.2f' % (correct_num / float(test_num)))