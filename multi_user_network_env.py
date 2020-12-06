import numpy as np
import random
import sys
import os


TIME_SLOTS = 1
NUM_CHANNELS = 3
NUM_USERS = 5
ATTEMPT_PROB = 0.6
GAMMA = 0.90


# 注意诀窍：
# 1. sample的action数组：下标(加一)为user的编号，元素值为channel的编号(包括0)；
# 2. step返回的obj的列表前半部分：下标(加一)为user的编号，元组值前者为是否被分配channel，后者为reward；
# 3. step返回的obj的列表后半部分：下标(加一)为channel的编号，元组值前者为是否被分配channel，后者为reward；

class env_network:
    # 环境包括三个元素：
    # 1）num_users：用户数 -> state
    # 2) num_channels: 通道数 -> action
    # 3) attempt_prob:
    def __init__(self,num_users,num_channels,attempt_prob):
        self.ATTEMPT_PROB = attempt_prob
        self.NUM_USERS = num_users
        self.NUM_CHANNELS = num_channels
        self.REWARD = 1

        #self.channel_alloc_freq =
        # 动作 a -> 分配到哪一通道
        self.action_space = np.arange(self.NUM_CHANNELS+1)
        # （s，a）
        self.users_action = np.zeros([self.NUM_USERS],np.int32)
        # 状态 s -> 每个用户当前的状态
        self.users_observation = np.zeros([self.NUM_USERS],np.int32)

    def reset(self):
        pass

    # 由于是多个用户，一次选择多个动作a，分别为一个用户选择一个动作
    # 每个动作对应一个channel，包括0，不选择任何channel
    # 注意：用户和channel之间存在对应关系
    # 例如：x = [1,0,2] 表示用户user1，user2，user3分别选择了channel1，None，Channel2
    def sample(self):
        x =  np.random.choice(self.action_space,size=self.NUM_USERS)
        return x

    def step(self,action):
        #print
        # 用户和channel之间存在对应关系，故而用户数和channel数目是相等的
        assert (action.size) == self.NUM_USERS, "action and user should have same dim {}".format(action)
        channel_alloc_frequency = np.zeros([self.NUM_CHANNELS + 1],np.int32)  #0 for no channel access
        obs = []
        reward = np.zeros([self.NUM_USERS])
        j = 0
        for each in action:
            prob = random.uniform(0,1)
            # 如果采样的概率小于预定义的ATTEMPT_PROB，就将用户和action（以及channel）对应起来
            if prob <= self.ATTEMPT_PROB:
                self.users_action[j] = each  # action
                # 表示该channel已经被选择过一次了，频率加一
                channel_alloc_frequency[each]+=1
            j+=1

        # 遍历所有的channel，如果一个channel被两个及以上的用户选择，则重置为0
        for i in range(1,len(channel_alloc_frequency)):
            if channel_alloc_frequency[i] > 1:
                channel_alloc_frequency[i] = 0
        #print channel_alloc_frequency

        for i in range(len(action)):
            # 查看用户选择了哪个频道
            self.users_observation[i] = channel_alloc_frequency[self.users_action[i]]
            # 如果用户选择了一个可用的频道，则其观测值为1，回报r为1
            if self.users_action[i] ==0:   #accessing no channel
                self.users_observation[i] = 0
            if self.users_observation[i] == 1:
                reward[i] = 1
            obs.append((self.users_observation[i],reward[i]))
        residual_channel_capacity = channel_alloc_frequency[1:]
        residual_channel_capacity = 1-residual_channel_capacity
        obs.append(residual_channel_capacity)
        # setp的返回值obs的格式如下：
        # [(ACK1,REWARD1),(ACK2,REWARD2),(ACK3,REWARD3), ...,(ACKn,REWARDn), (CAP_CHANNEL1,CAP_CHANNEL2,...,CAP_CHANNEL_k)]
        # n 表示user数目， k 表示channel数目
        # 前面部分表示，每个用户是否被分配了channel，以及对应的reward；后面部分表示，每个channel当前是否可用
        # 例如，假如num_user=3, num_channel=2, obs=[(1, 1.0), (0, 0.0), (1, 1.0), array([0, 0], dtype=int32)], 则表示：
        #   user1被分配了channel且对应reward为1.0; user2未被分配channel且对应reward为0.0;user3被分配了channel且对应reward为1.0；两个channel都已经被占完成
        return obs