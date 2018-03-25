import tensorflow as tf

dim = 2
nb_hidden = 1
nb_outputs = 1
t = 0.001      #cold temperature (sharp slop of the sigmoid)
x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[0], [1], [1], [0]]
nb_hbridge = nb_hidden + dim
x = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None, 1])
w1 = tf.Variable(tf.random_uniform([dim, nb_hidden], -1, 1, seed=0))
w2 = tf.Variable(tf.random_uniform([nb_hbridge, nb_outputs], -1, 1,  seed=0))
b1 = tf.Variable(tf.zeros([nb_hidden]), name="Biases1")
b2 = tf.Variable(tf.zeros([nb_outputs]), name="Biases2")
z1 = tf.matmul(x, w1) + b1
hlayer = tf.sigmoid(z1/t)
hlayer1 = tf.concat([hlayer, x], 1)
z2 = tf.matmul(hlayer1, w2) + b2
out = tf.sigmoid(z2/t)
loss = tf.reduce_mean(tf.square(y_train - out))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # initialize variables
curr_w1, curr_b1, curr_w2, curr_b2, curr_hlayer, curr_out, curr_loss = \
    sess.run([w1, b1, w2, b2, hlayer1, out, loss], {x: x_train,  y: y_train})
print('Before the fix, loss =', curr_loss)
print(sess.run(out, {x: x_train}))

fix_w1 = tf.assign(w1, [[1.], [1.]])
fix_w2 = tf.assign(w2, [[-2.], [1.], [1.]])
fix_b1 = tf.assign(b1, [-1.5])
fix_b2 = tf.assign(b2, [-0.5])
sess.run([fix_w1, fix_b1, fix_w2, fix_b2])
curr_w1, curr_b1, curr_w2, curr_b2, curr_hlayer, curr_out, curr_loss = \
    sess.run([w1, b1, w2, b2, hlayer1, out, loss], {x: x_train,  y: y_train})
print('After the fix, loss=', curr_loss)
print(sess.run(out, {x: x_train}))



