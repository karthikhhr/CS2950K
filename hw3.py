import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
w_list = [5, 5, 1, 32]
b_list = [32]
w_conv1 = tf.Variable((tf.truncated_normal(w_list, stddev=0.1)))
b_conv1 = tf.Variable(tf.constant(0.1, shape=b_list))
x_image = tf.reshape(x, [-1, 28, 28, 1])
conv = tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(conv + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
w_list = [5, 5, 32, 64]
b_list = [64]
w_conv2 = tf.Variable((tf.truncated_normal(w_list, stddev=0.1)))
b_conv2 = tf.Variable(tf.constant(0.1, shape=b_list))
conv =  tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(conv + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
w_list = [3136, 1024]
b_list = [1024]
w_fc1 = tf.Variable((tf.truncated_normal(w_list, stddev=0.1)))
b_fc1 = tf.Variable(tf.constant(0.1, shape=b_list))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
w_fc2 = tf.Variable((tf.truncated_normal([1024, 10], stddev=0.1)))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(2000):
    batch = mnist.train.next_batch(50)
    train_accuracy = accuracy.eval(feed_dict={
        x: batch[0], y_: batch[1], keep_prob: 1.0})
    print "training accuracy is " + str(train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print "test accuracy is" + str(accuracy.eval(feed_dict={
x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))