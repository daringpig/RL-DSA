import numpy as np
import  matplotlib.pyplot as plt

def get_states(batch):
    states = []
    for i in batch:
        states_per_batch = []
        for step_i in i:
            states_per_step = []
            for user_i in step_i[0]:
                states_per_step.append(user_i)
            states_per_batch.append(states_per_step)
        states.append(states_per_batch)

    return states


def get_actions(batch):
    actions = []
    for each in batch:
        actions_per_batch = []
        for step_i in each:
            actions_per_step = []
            for user_i in step_i[1]:
                actions_per_step.append(user_i)
            actions_per_batch.append(actions_per_step)
        actions.append(actions_per_batch)

    return actions


def get_rewards(batch):
    rewards = []
    for each in batch:
        rewards_per_batch = []
        for step_i in each:
            rewards_per_step = []
            for user_i in step_i[2]:
                rewards_per_step.append(user_i)
            rewards_per_batch.append(rewards_per_step)
        rewards.append(rewards_per_batch)
    return rewards


def get_next_states(batch):
    next_states = []
    for each in batch:
        next_states_per_batch = []
        for step_i in each:
            next_states_per_step = []
            for user_i in step_i[3]:
                next_states_per_step.append(user_i)
            next_states_per_batch.append(next_states_per_step)
        next_states.append(next_states_per_batch)
    return next_states


def get_states_user(batch, num_user):
    states = []
    for user in range(num_user):
        states_per_user = []
        for each in batch:
            states_per_batch = []
            for step_i in each:

                try:
                    states_per_step = step_i[0][user]

                except IndexError:
                    print(step_i)
                    print("-----------")

                    print("eror")

                    '''for i in batch:
                        print i
                        print "**********"'''
                    sys.exit()
                states_per_batch.append(states_per_step)
            states_per_user.append(states_per_batch)
        states.append(states_per_user)
    # print len(states)
    return np.array(states)


def get_actions_user(batch, num_user):
    actions = []
    for user in range(num_user):
        actions_per_user = []
        for each in batch:
            actions_per_batch = []
            for step_i in each:
                actions_per_step = step_i[1][user]
                actions_per_batch.append(actions_per_step)
            actions_per_user.append(actions_per_batch)
        actions.append(actions_per_user)
    return np.array(actions)


def get_rewards_user(batch, num_user):
    rewards = []
    for user in range(num_user):
        rewards_per_user = []
        for each in batch:
            rewards_per_batch = []
            for step_i in each:
                rewards_per_step = step_i[2][user]
                rewards_per_batch.append(rewards_per_step)
            rewards_per_user.append(rewards_per_batch)
        rewards.append(rewards_per_user)
    return np.array(rewards)


#
def get_next_states_user(batch, num_user):
    next_states = []
    for user in range(num_user):
        next_states_per_user = []
        for each in batch:
            next_states_per_batch = []
            for step_i in each:
                next_states_per_step = step_i[3][user]
                next_states_per_batch.append(next_states_per_step)
            next_states_per_user.append(next_states_per_batch)
        next_states.append(next_states_per_user)
    return np.array(next_states)


def draw_res(time_step, cum_collision, cum_r):
    if time_step % 5000 == 4999:
        plt.figure(1)

        plt.subplot(211)
        # plt.plot(np.arange(1000),total_rewards,"r+")
        # plt.xlabel('Time Slots')
        # plt.ylabel('total rewards')
        # plt.title('total rewards given per time_step')
        # plt.show()
        plt.plot(np.arange(5001), cum_collision, "r-")
        plt.xlabel('Time Slot')
        plt.ylabel('cumulative collision')
        # plt.show()

        plt.subplot(212)
        plt.plot(np.arange(5001), cum_r, "r-")
        plt.xlabel('Time Slot')
        plt.ylabel('Cumulative reward of all users')
        # plt.title('Cumulative reward of all users')

        plt.show()

        total_rewards = []
        cum_r = [0]
        cum_collision = [0]
        #saver.save(sess, 'checkpoints/dqn_multi-user.ckpt')
        # print time_step,loss , sum(reward) , Qs

    # print ("*************************************************")

def draw_res2(time_step, cum_collision, cum_r, loss_list):
    if time_step % 5000 == 4999:
        plt.figure(1)
        plt.subplot(311)
        # plt.plot(np.arange(1000),total_rewards,"r+")
        # plt.xlabel('Time Slots')
        # plt.ylabel('total rewards')
        # plt.title('total rewards given per time_step')
        # plt.show()
        plt.plot(np.arange(5001), cum_collision, "r-")
        plt.xlabel('Time Slot')
        plt.ylabel('cumulative collision')

        plt.subplot(312)
        plt.plot(np.arange(5001), cum_r, "r-")
        plt.xlabel('Time Slot')
        plt.ylabel('Cumulative reward of all users')


        plt.subplot(321)
        plt.plot(np.arange(len(loss_list)), loss_list, "b-")
        plt.xlabel('Time Slot')
        plt.ylabel('Loss')

        plt.show()

        total_rewards = []
        cum_r = [0]
        cum_collision = [0]
        #saver.save(sess, 'checkpoints/dqn_multi-user.ckpt')
        #print(time_step, loss, sum(reward), Qs)

    # print ("*************************************************")
