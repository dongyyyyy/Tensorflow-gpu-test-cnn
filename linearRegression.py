import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# try to find values for w and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # -1 ~ 1 사이의 난수
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # -1 ~ 1 사이의 난수

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# my hypothesis
hypothesis = W * X + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# cost = (((w*1+b) - 1) + ((w*2+b) - 2) - ((w*3+b) - 3))/3

# minimize
rate = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost) # train은 cost를 최소화시키는 것이 목표

# before starting, initialize the variables. We will 'run' this first.
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train,feed_dict={X:x_data,Y:y_data})
    if step % 20 == 0:
        print(step,sess.run(cost,feed_dict={X:x_data,Y:y_data}),sess.run(W),sess.run(b))

print(sess.run(hypothesis,feed_dict={X:5}))
