import tensorflow as tf
import numpy as np

#tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('result/' + 'result_csv_', delimiter=' ', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

def x_norm(x):
    x = np.array(x)
    x = x / 1000 * 0.999 + 0.001
    return x

def y_norm(y):
    y = np.array(y)
    y = y / 10000 * 0.9999 + 0.0001
    return y

def return_y_norm(y):
    y = (y - 0.0001) /0.9999 * 10000

x_data = x_norm(x_data)
y_data = y_norm(y_data)

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="Weight")
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="Bias")
print(b)

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

hypothesis = tf.add(tf.multiply(W, X), b)
print(hypothesis)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0004)
cost = tf.reduce_mean(tf.square(Y - hypothesis))
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(W), sess.run(b))

    for step in range(20000):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
        print("Step: ", step, "  Cost: ", cost_val)
        #print("Step: ", step, "  Cost: ", cost_val, "  W: ", sess.run(W), "  b: ", sess.run(b))

    print("X: 488, Y:", sess.run(hypothesis, feed_dict={X: x_norm(380)}))
