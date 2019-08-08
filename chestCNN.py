import tensorflow as tf
import numpy as np
from PIL import Image
import os

batch_size = 32
test_size = 64


def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.01))


def init_bias(shape):
  """bias_variable 주어진 shape에 대한 bias variable을 생성한다."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def load_image():
    path_train_normal = './dataset/chest/train/NORMAL/'
    path_train_pneumonia = './dataset/chest/train/PNEUMONIA/'
    path_val_normal = './dataset/chest/val/NORMAL'
    path_val_pneumonia = './dataset/chest/val/PNEUMONIA'
    file_list = os.listdir(path_train_normal) + os.listdir(path_train_pneumonia)
    test_list = os.listdir(path_val_normal) + os.listdir(path_val_pneumonia)
    train_list = []
    val_list = []
    train_label = [[0. for i in range(2)] for j in range(len(file_list))]
    val_label = [[0. for i in range(2)] for j in range(len(test_list))]
    i = 0
    for item in file_list:
        if item.find('.jpeg') >= 0 and item.find('person') >= 0: # 폐렴사진
            train_list.append(path_train_pneumonia+item)
            train_label[i][0] = 0.
            train_label[i][1] = 1.
            i += 1
        elif item.find('jpeg') >= 0 : # 정상사진
            train_list.append(path_train_normal+item)
            train_label[i][0] = 1.
            train_label[i][1] = 0.
            i += 1
    i = 0
    for item in test_list:
        if item.find('.jpeg') >= 0 and item.find('person') >= 0: # 폐렴사진
            val_list.append(path_val_pneumonia+item)
            val_label[i][0] = 0.
            val_label[i][1] = 1.
            i += 1
        elif item.find('jpeg') >= 0 : # 정상사진
            val_list.append(path_val_normal + item)
            val_label[i][0] = 1.
            val_label[i][1] = 0.
            i += 1
    return train_list, train_label , val_list, val_label


def batch_norm_cnn(batch_image,depth) :
    epsilon = 1e-5
    beta = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=True)
    mean, variance = tf.nn.moments(batch_image, axes=[0, 1, 2])
    #norm_batch = tf.layers.batch_normalization(batch_image,training)
    norm_batch = tf.nn.batch_normalization(batch_image, mean, variance, beta, gamma, epsilon)

    return norm_batch


def block(x,weight,filters):
    out = tf.nn.relu(tf.nn.conv2d(x,weight,strides=[1,1,1,1],padding='SAME'))
    out = batch_norm_cnn(out,filters)
    out = tf.nn.relu(out)
    out = tf.nn.max_pool2d(out,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    return out


def resnet_1(x):
    w = init_weights([7,7,1,64])
    b = init_bias([64])
    out = tf.nn.relu(tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME'))+b
    out = batch_norm_cnn(out,64)
    out = tf.nn.relu(out)
    return out


def res_block(x,in_filter,out_filter,stride=1): # 64 / 128
    identity = x
    if in_filter == out_filter:
        in_weight = init_weights([3,3,in_filter,out_filter]) # 64 64
        out_weight = init_weights([3,3,in_filter,out_filter]) # 64 64
    else:
        in_weight = init_weights([3, 3, in_filter, out_filter]) # 64 128
        out_weight = init_weights([3, 3, out_filter, out_filter]) # 128 128

    b1 = init_bias([out_filter])
    out = tf.nn.conv2d(x,in_weight,strides=[1,stride,stride,1],padding='SAME')+b1
    out = batch_norm_cnn(out,out_filter)
    out = tf.nn.relu(out)

    b2 = init_bias([out_filter])
    out = tf.nn.conv2d(out,out_weight,strides=[1,1,1,1],padding='SAME')+b2
    out = batch_norm_cnn(out,out_filter)
    if in_filter != out_filter:
        identity = tf.nn.conv2d(identity,in_weight,strides=[1,2,2,1],padding='SAME')
        identity = batch_norm_cnn(identity,out_filter)

    out += identity
    out = tf.nn.relu(out)

    return out


def resnet_model(x):
    print("x : ",x.shape) # 112 112 ->
    l1 = resnet_1(x)
    print(l1.shape)
    l1 = tf.nn.max_pool2d(l1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME') # 104 104 64
    print("maxpool : ",l1.shape)
    res1_1 = res_block(l1, 64, 64, stride=1) # 56 56 64 ->  104 104 64
    print("res1_1 : ", res1_1.shape)
    res1_2 = res_block(res1_1, 64, 64, stride= 1)
    print("res1_2 : ", res1_2.shape)
    res2_1 = res_block(res1_2, 64, 128, stride=2) # 28 28 128 -> 52 52 128
    print("res2_1 : ", res2_1.shape)
    res2_2 = res_block(res2_1, 128, 128, stride=1)
    print("res2_2 : ", res2_2.shape)

    res3_1 = res_block(res2_2, 128, 256, stride=2) # 14 14 256 -> 26 26 256
    print("res3_1 : ", res3_1.shape)
    res3_2 = res_block(res3_1, 256, 256, stride=1)
    print("res3_2 : ", res3_2.shape)

    res4_1 = res_block(res3_2, 256, 512, stride=2) # 7 7 512 -> 13 13 512
    print("res4_1 : ", res4_1.shape)
    res4_2 = res_block(res4_1, 512, 512, stride=1)
    print("res4_2 : ", res4_2.shape)

    out = tf.nn.avg_pool2d(res4_2, ksize=[1, 13, 13, 1], strides=[1, 13, 13, 1], padding='SAME')  # avg_pool
    #out = tf.nn.avg_pool(res4_2, ksize=[1,7,7,1],strides=[1,7,7,1],padding='SAME') # avg_pool
    print("avg pool : ",out.shape)
    out = tf.reshape(out, [-1, 512])  # flatten (?,512)
    print("flatten : " , out.shape)

    w_fc = init_weights([512,2])
    #b_result = init_bias([2])
    result = tf.nn.relu(tf.matmul(out, w_fc))
    #result = tf.nn.relu(tf.matmul(out, w_fc) + b_result)
    print("result : ",result.shape)
    return tf.nn.softmax(result) # 2개의 classification으로 구별 & softmax


def model(x,w1,w2,w3,w4,w0,b_l1,b_l2,b_l3,b_fc1,b_result,keep_conv = 0.7,keep_hidden = 0.5):
    '''
    l1 = tf.nn.relu(tf.nn.conv2d(input,w1,strides=[1,1,1,1],padding='SAME')+b_l1)# 224 224 32
    l1 = tf.nn.max_pool(l1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME') # 112 112 32
    l1 = tf.nn.dropout(l1,keep_prob=keep_conv)
    '''
    l1 = block(x,w1,32)
    print("l1 shape : ",l1.shape)
    '''
    l2 = tf.nn.relu(tf.nn.conv2d(l1,w2,strides=[1,1,1,1],padding='SAME')+b_l2) # 112 112 64
    l2 = tf.nn.max_pool(l2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME') # 56 56 64
    l2 = tf.nn.dropout(l2, keep_prob=keep_conv)
    '''
    l2 = block(l1,w2,64)
    print("l2 shape : ",l2.shape)

    '''
    l3 = tf.nn.relu(tf.nn.conv2d(l2,w3,strides=[1,1,1,1],padding='SAME')+b_l3) # 56 56 128
    l3 = tf.nn.max_pool(l3,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME') # 28 28 128
    l3 = tf.nn.dropout(l3, keep_prob=keep_conv) # 100
    l3 = tf.reshape(l3, [-1, 28 * 28 * 128]) #flatten
    '''
    l3 = block(l2,w3,128)
    l3 = tf.reshape(l3, [-1, 28 * 28 * 128]) #flatten
    print("l3 shape : ",l3.shape)

    fc1 = tf.nn.relu(tf.matmul(l3, w4) + b_fc1) # ? , 384
    fc1 = tf.nn.dropout(fc1,keep_prob=keep_hidden)
    print("fc shape : ",fc1.shape)
    result = tf.matmul(fc1, w0) + b_result
    #result = tf.nn.softmax(tf.nn.relu(tf.matmul(fc1,w0)+b_result))
    '''
    l4 = tf.nn.relu(tf.matmul(l3,w4))
    l4 = tf.nn.dropout(l4,keep_prob=keep_hidden)
    print("l4 shape : ",l4.shape)
    result = tf.matmul(l4,w_0)
    '''
    print("result shape: ",result.shape)
    return tf.nn.softmax(result)


def read_image(path):
    image = Image.open(path)
    image = image.resize((416, 416))
    #image = image.resize((224,224))
    image = np.array(image)
    # Channel 1을 살려주기 위해 reshape 해줌
    return image.reshape(image.shape[0], image.shape[1], 1) # grayscale 유지


def main():
    train_lists, train_label, val_lists , val_label = load_image()
    train_list = []
    val_list = []
    for item in train_lists:
        try:
            image = read_image(item)
            '''
            for i in range(224):
                for j in range(224):
                    image[i][j][1] = image[i][j][1]/255.
            '''
            for i in range(416):
                for j in range(416):
                    image[i][j][1] = image[i][j][1] / 255.

            print(image.shape)
        except:
            pass
        train_list.append(image)
    for item in val_lists:
        try:
            image = read_image(item)
            '''
            for i in range(224):
                for j in range(224):
                    image[i][j][1] = image[i][j][1]/255.
            '''
            for i in range(416):
                for j in range(416):
                    image[i][j][1] = image[i][j][1] / 255.
        except:
            pass
        val_list.append(image)

    print('Complete load image and label')
    print('=' * 20)
    print(train_lists[0])
    print(train_label[0])
    print('=' * 20)
    print(train_lists[1000])
    print(train_label[1000])
    print('=' * 20)
    print(train_lists[2000])
    print(train_label[2000])
    print('=' * 20)
    print(train_lists[3000])
    print(train_label[3000])
    print('=' * 20)
    print(train_lists[4000])
    print(train_label[4000])
    print('='*20)
    X = tf.placeholder("float", [None, 416, 416, 1])
    #X = tf.placeholder("float",[None,224,224,1])
    Y = tf.placeholder("float",[None,2])
    #training = tf.placeholder("float",[None,2])
    '''
    w1 = init_weights([3, 3, 1, 32])
    w2 = init_weights([3, 3, 32, 64])
    w3 = init_weights([3, 3, 64, 128])
    w4 = init_weights([28*28*128, 384])
    w0 = init_weights([384,2])

    b_l1 = init_bias([32])
    b_l2 = init_bias([64])
    b_l3 = init_bias([128])
    b_fc1 = init_bias([384])
    b_result = init_bias([2])

    keep_conv = tf.placeholder("float")
    keep_hidden = tf.placeholder("float")
    '''
    #trains = model(X,w1,w2,w3,w4,w0,b_l1,b_l2,b_l3,b_fc1,b_result,keep_conv,keep_hidden)
    trains = resnet_model(X)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=trains, labels=Y))
    #cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(trains), axis=1))
    #train_op = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=trains,labels=Y))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=trains, labels=Y))
    #train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    predict_op = tf.argmax(trains,1)

    #correct_prediction = tf.equal(predict_op)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('start training')
        for epoch in range(20):
            avg_cost = 0
            training_batch = len(train_list)//batch_size
            start = 0
            end = len(train_list)
            for j in range(training_batch):
                if start+batch_size <= len(train_list):
                    batch_xs = np.array(train_list[start:start+batch_size])
                    batch_ys = np.array(train_label[start:start+batch_size])
                    #batch_ys = batch_ys.reshape(-1,2)
                    start = start+batch_size
                else:
                    batch_xs = np.array(train_list[start:len(train_list)])
                    batch_ys = np.array(train_label[start:len(train_list)])
                    #batch_ys = batch_ys.reshape(-1,2)
                feed_dict = {X: batch_xs, Y: batch_ys}
                #feed_dict = {X: batch_xs , Y:batch_ys,keep_conv:0.7 , keep_hidden: 0.5}
                c, _ = sess.run([cost,train_op],feed_dict=feed_dict)
                avg_cost += c / training_batch
                print("batch cost : ",c)
            print('Epoch : ','%04d' % (epoch+1),'cost : ','{:.9f}'.format(avg_cost))
        print('Learning Finished')


        for val in range(0,len(val_list)):
            #feed_dict={X:np.array(val_list[val]).reshape(-1,224,224,1),keep_conv:0.7 , keep_hidden: 0.5}
            feed_dict = {X: np.array(val_list[val]).reshape(-1, 416, 416, 1)}
            #feed_dict = {X: np.array(val_list[val]).reshape(-1, 224, 224, 1)}
            print("val = ",val,"val_lists = ",val_lists[val] , "Image = ", val_label[val],"Predict = ",sess.run(trains, feed_dict=feed_dict))


main()