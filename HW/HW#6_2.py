#### HW#6   cliff walking
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

class Cliff_Walking():
    def __init__(self):
        super(Cliff_Walking, self).__init__()
        #### action 0: up, 1: right, 2: down, 3: left
        self.ACTION = [(0,1), (1,0), (0,-1), (-1,0)]
        self.DISCOUNT_FACTOR = 1
        self.EPS = 0.1
        self.ALPHA = 0.5
        self.LAMBDA = 0.9
        self.STEP_REWARD = -1

        self.init_state = np.array([0,0])
        self.Q_TABLE = np.zeros((12,4,4))        #### x y action
        self.Q_TABLE_ = np.zeros((12,4,4))
        self.E_TABLE = np.zeros((12,4,4))

    def ALL_INIT(self):
        self.Q_TABLE = np.zeros((12, 4, 4))  #### x y action
        self.Q_TABLE_ = np.zeros((12, 4, 4))
        self.E_TABLE = np.zeros((12, 4, 4))

    def Q_learning(self, repeat=1):
        self.ALL_INIT()
        print("Start Q learning")
        plot_reward = [0]*500
        for i in range(repeat):
            self.ALL_INIT()
            if i % 5 == 4:
                print(i)
            for episode in range(500):
                reward = self.Q_learning_update()
                plot_reward[episode] += reward
        plot_reward = np.array(plot_reward) / 100
        if repeat != 1:
            plt.plot(np.array(plot_reward))

    def test_Q_learning(self):
        print("Test Q learning")
        plot_reward = []
        for episode in range(500):
            reward = self.Q_learning_update(greedy = False, update = False)
            plot_reward += [reward]
        plt.plot(np.array(plot_reward))


    def Q_learning_update(self, greedy = True, update = True):
        time_step = 0
        end_flag = 0
        state = copy.deepcopy(self.init_state)
        total_reward = 0
        while True:
            if time_step % 1000 == 999:
                print("time step = {:05d}".format(time_step+1))
            reward = -1
            time_step += 1
            state_ = copy.deepcopy(state)
            action = self.choose_action(state, greedy)

            #### by action
            state_ += self.ACTION[action]

            #### wall
            state_[0] = min(max(state_[0], 0), 11)
            state_[1] = min(max(state_[1], 0), 3)

            if (0 < state_[0] < 11) and state_[1] == 0:
                reward = -100
                state_ = self.init_state
                end_flag = 1

            #### Q table update
            if update:
                self.Q_TABLE_ = copy.deepcopy(self.Q_TABLE)
                self.Q_TABLE_[state[0], state[1], action] += self.ALPHA * (reward + self.DISCOUNT_FACTOR * max(self.Q_TABLE[state_[0], state_[1]]) - self.Q_TABLE[state[0], state[1], action])

                self.Q_TABLE = copy.deepcopy(self.Q_TABLE_)
            state = copy.deepcopy(state_)

            total_reward += reward
            if (state_[0] == 11 and state_[1] == 0) or end_flag == 1:
                # print(total_reward)
                return total_reward

    def SARSA(self, repeat=1):
        self.ALL_INIT()
        print("Start SARSA")
        plot_reward = [0]*500
        for i in range(repeat):
            self.ALL_INIT()
            if i % 5 == 4:
                print(i)
            for episode in range(500):
                reward = self.SARSA_update()
                plot_reward[episode] += reward
        plot_reward = np.array(plot_reward) / 100
        if repeat != 1:
            plt.plot(np.array(plot_reward))


    def test_SARSA(self):
        print("Test SARSA")
        plot_reward = []
        for episode in range(500):
            reward = self.SARSA_update(greedy = False, update = False)
            plot_reward += [reward]
        plt.plot(np.array(plot_reward))


    def SARSA_update(self, greedy = True, update = True):
        time_step = 0
        end_flag = 0
        state = copy.deepcopy(self.init_state)
        total_reward = 0
        while True:
            if time_step % 1000 == 999:
                print("time step = {:05d}".format(time_step+1))
            reward = -1
            time_step+=1
            state_ = copy.deepcopy(state)
            action = self.choose_action(state, greedy)

            #### by action
            state_ += self.ACTION[action]

            #### wall
            state_[0] = min(max(state_[0], 0), 11)
            state_[1] = min(max(state_[1], 0), 3)

            if (0< state_[0] < 11) and state_[1] == 0:
                reward = -100
                state_ = self.init_state
                end_flag = 1

            action_ = self.choose_action(state_)

            #### Q table update
            if update:
                self.Q_TABLE_ = copy.deepcopy(self.Q_TABLE)
                self.Q_TABLE_[state[0], state[1], action] += self.ALPHA * (reward + self.DISCOUNT_FACTOR * self.Q_TABLE[state_[0], state_[1], action_] - self.Q_TABLE[state[0], state[1], action])

                self.Q_TABLE = copy.deepcopy(self.Q_TABLE_)
            state = copy.deepcopy(state_)

            total_reward += reward
            if (state_[0] == 11 and state_[1] == 0) or end_flag == 1:
                # print(total_reward)
                return total_reward


    def choose_action(self, state, greedy=True):
        if (random.random() < self.EPS) and greedy:
            #### exploration
            return random.randint(0,3)          #### randomly choose 0~3 action

        else:
            Q_values = self.Q_TABLE[state[0], state[1]]
            max_val = Q_values.max()
            #### if max_val has multiple max value, choose randomly
            return np.random.choice(np.where(Q_values == max_val)[0])





if __name__ == '__main__':
    PLAY = Cliff_Walking()
    PLAY.SARSA()
    PLAY.test_SARSA()
    PLAY.Q_learning()
    PLAY.test_Q_learning()
    plt.legend(["SARSA", "Q learning"])
    plt.axis([0, 500, -100, 0])
    plt.savefig("./Cliff_Walking_axis.png")


