import tensorflow as tf

dim = 2
nb_outputs = 1
nb_hidden = 4
temp = 0.001
x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[0], [1], [1], [0]]
x = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None, 1])
w1 = tf.Variable(tf.random_uniform([dim, nb_hidden], -1, 1, seed=0),     name="Weights1")
w2 = tf.Variable(tf.random_uniform([nb_hidden, nb_outputs], -1, 1,   seed=0), name="Weights2")
b1 = tf.Variable(tf.zeros([nb_hidden]), name="Biases1")
b2 = tf.Variable(tf.zeros([nb_outputs]), name="Biases2")

z1 = tf.matmul(x, w1) + b1  # [4,2]X[2,nb_hidden]  results in a vector [4,nb_hidden] of z’s
hlayer1 = tf.sigmoid(z1/temp)  # elemen wise
z2 = tf.matmul(hlayer1, w2) + b2
out = tf.sigmoid(z2/temp)
loss = tf.reduce_mean(tf.square(y_train - out))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print('Before the fix:')
print(sess.run(out, {x: x_train}))
print('Loss =', sess.run(loss, {x: x_train}))


fix_w1 = tf.assign(w1, [[1., 1., 1., 1.], [1., 1., 1., 1.]])
fix_w2 = tf.assign(w2, [[-1.], [1.], [1.], [-1.]])
fix_b1 = tf.assign(b1, [.5, -0.5, -0.5, -1.5])
fix_b2 = tf.assign(b2, [-0.5])

sess.run([fix_w1, fix_b1, fix_w2, fix_b2])

print('After the fix:')
print(sess.run(out, {x: x_train}))
print('Loss =', sess.run(loss, {x: x_train}))
