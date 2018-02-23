from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import time
import tensorflow as tf
sess = tf.InteractiveSession()

in_units = 784
hi_units = 300
x = tf.placeholder(tf.float32,[None,in_units])
y_ = tf.placeholder(tf.float32,[None,10])

W1 = tf.Variable(tf.truncated_normal([in_units,hi_units],stddev=0.1))
b1 = tf.Variable(tf.zeros([hi_units]))
W2 = tf.Variable(tf.zeros([hi_units,10]))
b2 = tf.Variable(tf.zeros([10]))


keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y_conv = tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)


#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
#train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
#
#tf.global_variables_initializer().run()
#
#for i in range(3000):
#	batch_xs,batch_ys = mnist.train.next_batch(100)
#	train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i % 100 == 0:
			train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})#accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
			print("step %d, training accuracy %g" % (i,train_accuracy))
		sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
		#train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

	print("test accuracy %g" % sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))#accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))

