import tensorflow as tf

x_data = [[1.,0.],[0.,2.],[3.,0.],[0.,4.],[5.,0.]]
y_data = [[1.],[2.],[3.],[4.],[5.]]
# try to find values for w and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_normal([2,1]),name='weight')
b = tf.Variable(tf.random_normal([1], name = 'bias')) # -1 ~ 1 사이의 난수

X = tf.placeholder(tf.float32,shape=[None,2])
#X2 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# my hypothesis
#hypothesis = W1 * X1 + W2 * X2 + b
hypothesis = tf.matmul(X, W) + b
# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# cost = (((w*1+b) - 1) + ((w*2+b) - 2) - ((w*3+b) - 3))/3

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost) # train은 cost를 최소화시키는 것이 목표

# launch the graph
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# fit the line
for step in range(2001):
    cost_val , hy_val , _ = sess.run([cost,hypothesis,train],feed_dict={X: x_data,Y:y_data})
    if step % 20 == 0:
        print(step,"Cost : ",cost_val,"\nPredict : ",hy_val )

print("Test = ", sess.run(hypothesis,feed_dict={X:[[1.,2.],[2.,3.],[3.,4.],[4.,5.],[6.,7.]]}))
