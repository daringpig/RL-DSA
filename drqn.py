import tensorflow as tf
import numpy as np

# -- 整个网络输入包括三个，即states，actions和targetQs
# -- 网络的中间输出是预测的actions，即推导映射s->Q(s,a), 虽然看起来像是s->a
# 训练的目标是 loss = (推导出的Q - 输入的targetQ), 并进行优化
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, 
                 action_size=2, hidden_size=10, step_size=1 ,
                 name='QNetwork'):
        
        with tf.variable_scope(name):

            # -- 输入包括三个，即states，actions和targetQs
            # 输入states
            # size=[batchsize, step_size, state_size]，step_size相当于序列长度，state_size相当于特征数
            self.inputs_ = tf.placeholder(tf.float32, [None,step_size, state_size], name='inputs_')
            # 输入actions
            # 为方便后面的处理需要将actions改为one-hot编码
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            with tf.device('/cpu:0'):
                one_hot_actions = tf.one_hot(self.actions_, action_size)
            # 输入targetQs_
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            ##########################################

            #-----------------
            print("Shape of inputs_: ", self.inputs_.shape)
            print("Shape of actions_: ", self.actions_.shape)
            print("Shape of one_hot_actions: ", one_hot_actions.shape)
            print("Shape of targetQs_: ", self.targetQs_.shape)
         
            self.lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            
            self.lstm_out, self.state = tf.nn.dynamic_rnn(self.lstm,self.inputs_,dtype=tf.float32)

            # 取输出序列中的最后一个值，得到lstm的输出
            self.reduced_out = self.lstm_out[:,-1,:]
            self.reduced_out = tf.reshape(self.reduced_out,shape=[-1,hidden_size])

            print('Shape of lstm_out: ', self.lstm_out.shape)
            print('Shape of reduced_out: ', self.reduced_out.shape)

            #########################################
            
            #self.w1 = tf.Variable(tf.random_uniform([state_size,hidden_size]))
            #self.b1 = tf.Variable(tf.constant(0.1,shape=[hidden_size]))
            #self.h1 = tf.matmul(self.inputs_,self.w1) + self.b1
            #self.h1 = tf.nn.relu(self.h1)
            #self.h1 = tf.contrib.layers.layer_norm(self.h1)
            #'''

            self.w2 = tf.Variable(tf.random_uniform([hidden_size, hidden_size]))
            self.b2 = tf.Variable(tf.constant(0.1,shape=[hidden_size]))
            self.h2 = tf.matmul(self.reduced_out,self.w2) + self.b2
            self.h2 = tf.nn.relu(self.h2)
            self.h2 = tf.contrib.layers.layer_norm(self.h2)
            print('Shape of h2: ', self.h2.shape)

            self.w3 = tf.Variable(tf.random_uniform([hidden_size, action_size]))
            self.b3 = tf.Variable(tf.constant(0.1,shape=[action_size]))
            self.output = tf.matmul(self.h2,self.w3) + self.b3
            print('Shape of output: ', self.output.shape)

            #self.output = tf.contrib.layers.layer_norm(self.output)
           

            '''
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)
           
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size,activation_fn=None)            
            '''

            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            print('Shape of Q: ', self.Q.shape)
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            print('Shape of loss: ', self.Q.shape)
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)





from collections import deque

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size,step_size):
        idx = np.random.choice(np.arange(len(self.buffer)-step_size), 
                               size=batch_size, replace=False)
        
        res = []                       
                             
        for i in idx:
            temp_buffer = []  
            for j in range(step_size):
                temp_buffer.append(self.buffer[i+j])
            res.append(temp_buffer)
        return res