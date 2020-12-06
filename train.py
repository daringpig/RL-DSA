from  multi_user_network_env import env_network
from drqn import QNetwork,Memory
from utils import get_states_user, get_actions_user, get_rewards_user, get_next_states_user
from utils import draw_res, draw_res2
import numpy as np
import sys
import  matplotlib.pyplot as plt 
from collections import  deque
import os
import tensorflow as tf
import time

###################################################################
# ---- 参数设定

# 运行的DSA的次数
TIME_SLOTS = 100000                            # number of time-slots to run simulation
# 信道数目
NUM_CHANNELS = 2                               # Total number of channels
# 用户数目
NUM_USERS = 3                                  # Total number of users
ATTEMPT_PROB = 1                               # attempt probability of ALOHA based models

memory_size = 1000                      #size of experience replay deque
batch_size = 6                          # Num of batches to train at each time_slot
pretrain_length = batch_size            #this is done to fill the deque up to batch size before training
hidden_size = 128                       #Number of hidden neurons
learning_rate = 0.0001                  #learning rate
explore_start = .02                     #initial exploration rate
explore_stop = 0.01                     #final exploration rate
decay_rate = 0.0001                     #rate of exponential decay of exploration
gamma = 0.9                             #discount  factor
noise = 0.1
step_size=1+2+2                         #length of history sequence for each datapoint in batch
state_size = 2 *(NUM_CHANNELS + 1)      #length of input (2 * k + 2) : k = NUM_CHANNELS
action_size = NUM_CHANNELS+1            #length of output  (k+1)
alpha=0                                 #co-operative fairness constant
beta = 1                                #Annealing constant for Monte - Carlo

###################################################################

#It creates a one hot vector of a number as num with size as len
def one_hot(num,len):
    assert num >=0 and num < len ,"error"
    vec = np.zeros([len],np.int32)
    vec[num] = 1
    return vec

# 状态state生成器，相当于与环境交互后获得状态，实现映射(s,a)->(s_,r)
#generates next-state from action and observation
def state_generator(action,obs):
    input_vector = []
    if action is None:
        print ('None')
        sys.exit()
    # 计算每一个用户下一时刻的状态s
    for user_i in range(action.size):
        # action的one-hot向量
        input_vector_i = one_hot(action[user_i],NUM_CHANNELS+1)
        # channel向量
        channel_alloc = obs[-1]
        input_vector_i = np.append(input_vector_i,channel_alloc)
        input_vector_i = np.append(input_vector_i,int(obs[user_i][0]))    #ACK
        input_vector.append(input_vector_i)
    return input_vector

###################################################################
# ---- 初始化

# reseting default tensorflow computational graph
tf.reset_default_graph()

# initializing the environment
env = env_network(NUM_USERS,NUM_CHANNELS,ATTEMPT_PROB)

# initializing deep Q network
mainQN = QNetwork(name='main',hidden_size=hidden_size,learning_rate=learning_rate,step_size=step_size,state_size=state_size,action_size=action_size)

# this is experience replay buffer(deque) from which each batch will be sampled and fed to the neural network for training
memory = Memory(max_size=memory_size)   

# this is our input buffer which will be used for predicting next Q-values
history_input = deque(maxlen=step_size)

# to sample random actions for each user
action  =  env.sample()
obs = env.step(action)
state = state_generator(action,obs)
reward = [i[1] for i in obs[:NUM_USERS]]

###################################################################
# ---- 为训练准备离线数据

for ii in range(pretrain_length*step_size*5):
    action = env.sample()
    obs = env.step(action)      # obs is a list of tuple with [[(ACK,REW) for each user] ,CHANNEL_RESIDUAL_CAPACITY_VECTOR]
    next_state = state_generator(action,obs)
    reward = [i[1] for i in obs[:NUM_USERS]]
    memory.add((state,action,reward,next_state))
    state = next_state
    history_input.append(state)

###################################################################
# ---- 准备网络的训练

interval = 1       # debug interval

# saver object to save the checkpoints of the DQN to disk
saver = tf.train.Saver()

#initializing the session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)

#initialing all the tensorflow variables
sess.run(tf.global_variables_initializer())


#list of total rewards
total_rewards = []

# cumulative reward
cum_r = [0]

# cumulative collision
cum_collision = [0]

loss_list = []

##########################################################################
####                      main simulation loop                    ########

# 主循环：逐个slot执行
for time_step in range(TIME_SLOTS):
    
    # changing beta at every 50 time-slots
    if time_step %50 == 0:
        if time_step < 5000:
            beta -=0.001

    # exploration probability的值越来越大，但是增幅越来越小
    # current exploration probability
    explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate*time_step)
   

    # for test
    #print("\n\n================= In Timeslot %d =================" % time_step)


    # @@@@@@ 第一部分，执行policy得到action，开始执行映射s->a
    # -----------------------------------------------------
    # Exploration
    if explore_p > np.random.rand():
        #random action sampling
        action  = env.sample()
        #print("++++++++++++++++ Explored ++++++++++++++++")

    # Exploitation
    else:
        # initializing action vector
        action = np.zeros([NUM_USERS],dtype=np.int32)

        # converting input history into numpy array
        state_vector = np.array(history_input)

        # print np.array(history_input)
        #print("---------------- Exploited ----------------")

        for each_user in range(NUM_USERS):
            
            #feeding the input-history-sequence of (t-1) slot for each user seperately
            feed = {mainQN.inputs_:state_vector[:,each_user].reshape(1,step_size,state_size)}

            #predicting Q-values of state respectively
            Qs = sess.run(mainQN.output,feed_dict=feed) 
            #print Qs

            #   Monte-carlo sampling from Q-values  (Boltzmann distribution)
            ##################################################################################
            prob1 = (1-alpha)*np.exp(beta*Qs)

            # Normalizing probabilities of each action  with temperature (beta) 
            prob = prob1/np.sum(np.exp(beta*Qs)) + alpha/(NUM_CHANNELS+1)
            #print prob 

            #   This equation is as given in the paper :
            #   Deep Multi-User Reinforcement Learning for  
            #   Distributed Dynamic Spectrum Access :
            #   @Oshri Naparstek and Kobi Cohen (equation 12)
            ########################################################################################

            #  choosing action with max probability
            action[each_user] = np.argmax(prob,axis=1)

            #action[each_user] = np.argmax(Qs,axis=1)
            if time_step % interval == 0:
                #print (state_vector[:,each_user])
                #print (Qs)
                #print (prob, np.sum(np.exp(beta*Qs)))
                pass
    # @@@@@@ 第一部分，执行policy得到action，结束执行映射s->a
    # -----------------------------------------------------


    # @@@@@@ 第二部分，开始与环境交互，开始执行映射(s,a)->(s_,r)
    # -----------------------------------------------------

    # taking action as predicted from the q values and receiving the observation from the environment
    obs = env.step(action)           # obs is a list of tuple with [(ACK,REW) for each user ,(CHANNEL_RESIDUAL_CAPACITY_VECTOR)] 
    
    #print("Actions: \n", action)
    #print("States: \n", obs)

    # Generate next state from action and observation 
    next_state = state_generator(action,obs)
    #print("Next States: \n", next_state)

    # reward for all users given by environment
    reward = [i[1] for i in obs[:NUM_USERS]]
    
    # calculating sum of rewards
    sum_r =  np.sum(reward)

    #calculating cumulative reward
    cum_r.append(cum_r[-1] + sum_r)

    #If NUM_CHANNELS = 2 , total possible reward = 2 , therefore collision = (2 - sum_r) or (NUM_CHANNELS - sum_r) 
    collision = NUM_CHANNELS - sum_r
    
    #calculating cumulative collision
    cum_collision.append(cum_collision[-1] + collision)
    
   
    #############################
    #  for co-operative policy we will give reward-sum to each user who have contributed
    #  to play co-operatively and rest 0
    for i in range(len(reward)):
        if reward[i] > 0:
            reward[i] = sum_r
    #############################


    total_rewards.append(sum_r)
    #print("Rewards: \n", reward)

    # @@@@@@ 第二部分，开始与环境交互，结束执行映射(s,a)->(s_,r)
    # -----------------------------------------------------



    # @@@@@@ 第三部分，更新缓冲区，准备训练样本
    # -----------------------------------------------------
    
    # add new experiences into the memory buffer as (state, action , reward , next_state) for training
    memory.add((state,action,reward,next_state))
    
    
    state = next_state
    #add new experience to generate input-history sequence for next state
    history_input.append(state)

    # @@@@@@ 第四部分，训练模型，学习参数
    #  Training block starts
    ###################################################################################

    #  sampling a batch from memory buffer for training
    batch = memory.sample(batch_size,step_size)
    
    #   matrix of rank 4
    #   shape [NUM_USERS,batch_size,step_size,state_size]
    states = get_states_user(batch, NUM_USERS)
  
    #   matrix of rank 3
    #   shape [NUM_USERS,batch_size,step_size]
    actions = get_actions_user(batch, NUM_USERS)
    
    #   matrix of rank 3
    #   shape [NUM_USERS,batch_size,step_size]
    rewards = get_rewards_user(batch, NUM_USERS)
    
    #   matrix of rank 4
    #   shape [NUM_USERS,batch_size,step_size,state_size]
    next_states = get_next_states_user(batch, NUM_USERS)
    
    #   Converting [NUM_USERS,batch_size]  ->   [NUM_USERS * batch_size]  
    #   first two axis are converted into first axis
    # -------------- 得到s, a, s_, r --------------
    states = np.reshape(states,[-1,states.shape[2],states.shape[3]])
    actions = np.reshape(actions,[-1,actions.shape[2]])
    rewards = np.reshape(rewards,[-1,rewards.shape[2]])
    next_states = np.reshape(next_states,[-1,next_states.shape[2],next_states.shape[3]])

    #print(states.shape)

    # -------------- 得到q_target --------------
    #  creating target vector (possible best action)
    target_Qs = sess.run(mainQN.output,feed_dict={mainQN.inputs_:next_states})

    #  Q_target =  reward + gamma * Q_next
    targets = rewards[:,-1] + gamma * np.max(target_Qs,axis=1)
  
    #  calculating loss and train using Adam  optimizer 
    loss, _ = sess.run([mainQN.loss,mainQN.opt],
                            feed_dict={mainQN.inputs_:states,
                            mainQN.targetQs_:targets,
                            mainQN.actions_:actions[:,-1]})
    loss_list.append(loss)


    if time_step % 1000 == 0:
        print("------------------------1000---------------------")
    #draw_res(time_step, cum_collision, cum_r)
    draw_res2(time_step, cum_collision, cum_r, loss_list)
    saver.save(sess, 'checkpoints/dqn_multi-user.ckpt')
