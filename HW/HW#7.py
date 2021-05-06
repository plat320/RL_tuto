#### HW#7   Mountain Car
import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt

class Mountain_Car():
    def __init__(self):
        super(Mountain_Car, self).__init__()
        #### action 0: up, 1: right, 2: down, 3: left
        self.ACTION = [-1, 0, 1]
        self.DISCOUNT_FACTOR = 1
        self.EPS = 0.1
        self.ALPHA = 0.5 / 8
        self.LAMBDA = 0.9
        self.STEP_REWARD = -1

        self.num_tile = 8
        self.TILE_offset = [[(1*n) % (8-1), (3 * n) % (8-1)] for n in range(8)]

        self.W_TABLE = np.zeros((self.num_tile,self.num_tile,self.num_tile**2,3))            #### tile8, state64x64, action3

        self.max_vel = 0.07
        self.min_vel = -0.07
        self.unit_vel = (self.max_vel - self.min_vel)/(self.num_tile**2 - 7)
        self.max_pos = 0.5
        self.min_pos = -1.2
        self.unit_pos = (self.max_pos - self.min_pos)/(self.num_tile**2 - 7)


    def ALL_INIT(self):
        self.W_TABLE = np.zeros((self.num_tile,self.num_tile,self.num_tile,3))            #### tile8, state64x64, action3

    def init_state(self):
        return [random.uniform(-0.6, -0.4), 0]

    def cont2discrete_state(self, state):
        return [(state[0] - self.min_pos)//self.unit_pos,
                (state[1] - self.min_vel)//self.unit_vel]

    def get_feature_vector(self, state, action):
        feature_vector = np.zeros((8,8,8,3))
        discrete_state = self.cont2discrete_state(state)
        for i in range(8):
            x = int((discrete_state[0] + 7 - self.TILE_offset[i][0])//8)
            y = int((discrete_state[1] + 7 - self.TILE_offset[i][1])//8)
            feature_vector[i,x,y,action] += 1
        return feature_vector

    def choose_action(self, state, random_select=False, greedy=True):
        if ((random.random() < self.EPS) and greedy) or random_select:
            #### exploration
            return random.randint(0, 2)  #### randomly choose 0~2 action

        else:
            Q_value_list = []
            for action in self.ACTION:
                feature_vector = self.get_feature_vector(state, action)
                Q_value_list.append(np.multiply(feature_vector, self.W_TABLE).sum())

            max_val = max(Q_value_list)
            #### if max_val has multiple max value, choose randomly
            return np.random.choice(np.where(Q_value_list == max_val)[0])



    def Semi_grad_SARSA(self, repeat=1):
        print("Start semi gradient SARSA")
        plot_reward = [0]*500
        ALPHA = [0.1/8, 0.2/8, 0.5/8]
        for alpha in ALPHA:
            self.ALPHA = alpha
            for i in range(repeat):
                self.ALL_INIT()
                if i % 10 == 9:
                    print(i)
                for episode in range(500):
                    reward = self.Semi_grad_SARSA_update()
                    # print(reward)
                    plot_reward[episode] += reward

            plot_reward = np.array(plot_reward)/repeat
            plt.plot(np.array(plot_reward))
        # plt.show()
        # if repeat != 1:
        #     plt.plot(np.array(plot_reward))

    def Semi_grad_SARSA_update(self, greedy = True, update = True):
        time_step = 0
        total_reward = 0
        state = self.init_state()
        action = 0

        while True:
            if time_step % 10000 == 9999:
                print("time step = {:05d}".format(time_step+1))
            time_step += 1
            state_ = [0,0]

            #### by action
            state_[1] = max(min(state[1] + 0.001 * action - 0.0025 * math.cos(3*state[0]), 0.07), -0.07)
            state_[0] = max(min(state[0] + state_[1], 0.5), -1.2)

            #### get feature vector
            feature_vector = self.get_feature_vector(state, action)
            Q_value = np.multiply(feature_vector, self.W_TABLE).sum()

            if state_[0] == 0.5:
                # self.W_TABLE += self.ALPHA * (self.STEP_REWARD - Q_value) * feature_vector
                return total_reward

            action_idx_ = self.choose_action(state_, random_select=False, greedy=greedy)
            action_ = self.ACTION[action_idx_]

            #### get feature vector
            feature_vector_ = self.get_feature_vector(state_, action_)
            Q_value_ = np.multiply(feature_vector_, self.W_TABLE).sum()

            self.W_TABLE += self.ALPHA * (self.STEP_REWARD + self.DISCOUNT_FACTOR * Q_value_ - Q_value) * feature_vector

            state = copy.deepcopy(state_)
            action = copy.deepcopy(action_)
            total_reward += self.STEP_REWARD




if __name__ == '__main__':
    PLAY = Mountain_Car()
    PLAY.Semi_grad_SARSA(10)
    plt.legend(["0.1", "0.2", "0.5"])
    plt.axis([0, 500, -1000, 0])
    # plt.show()
    plt.savefig("./Mountain_car.png")


