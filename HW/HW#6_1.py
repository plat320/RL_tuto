#### HW#6 windy grid SARSA and SARSA_lambda
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

class Windy_Gridworld():
    def __init__(self):
        super(Windy_Gridworld, self).__init__()
        #### action 0: up, 1: right, 2: down, 3: left
        self.ACTION = [(0,1), (1,0), (0,-1), (-1,0)]
        self.DISCOUNT_FACTOR = 1
        self.EPS = 0.1
        self.ALPHA = 0.5
        self.LAMBDA = 0.9
        self.STEP_REWARD = -1

        self.init_state = np.array([0,3])
        self.Q_TABLE = np.zeros((10,7,4))        #### x y action
        self.Q_TABLE_ = np.zeros((10,7,4))
        self.E_TABLE = np.zeros((10,7,4))

    def ALL_INIT(self):
        self.Q_TABLE = np.zeros((10, 7, 4))  #### x y action
        self.Q_TABLE_ = np.zeros((10, 7, 4))
        self.E_TABLE = np.zeros((10,7,4))

    def SARSA_lambda(self):
        self.ALL_INIT()
        print("Start SARSA lambda")
        total_time_step = 0
        episode = 0
        plot_step = []
        while True:
            self.E_TABLE = np.zeros((10,7,4))
            time_step = self.SARSA_lambda_update()

            total_time_step += time_step
            plot_step += [total_time_step]
            if total_time_step > 8000:
                break
            episode +=1
        plt.plot(np.array(plot_step))



    def SARSA_lambda_update(self):
        time_step = 0
        state = copy.deepcopy(self.init_state)
        while True:
            if time_step % 1000 == 999:
                print("time step = {:05d}".format(time_step+1))
            time_step+=1
            state_ = copy.deepcopy(state)
            action = self.choose_action(state)

            #### by action
            state_ += self.ACTION[action]

            #### by wind
            if state[0] in [3,4,5,8]:
                state_[1] +=1
            elif state[0] in [6,7]:
                state_[1] +=2

            #### wall
            state_[0] = min(max(state_[0], 0), 9)
            state_[1] = min(max(state_[1], 0), 6)

            action_ = self.choose_action(state_)

            #### Q table update
            delta = self.STEP_REWARD + self.DISCOUNT_FACTOR * self.Q_TABLE[state_[0], state_[1], action_] - self.Q_TABLE[state[0], state[1], action]
            self.E_TABLE[state[0], state[1], action] +=1
            self.Q_TABLE[state[0], state[1], action] += self.ALPHA * delta * self.E_TABLE[state[0], state[1], action]
            self.E_TABLE *= self.DISCOUNT_FACTOR * self.LAMBDA

            state = copy.deepcopy(state_)

            if state_[0] == 7 and state_[1] == 3:
                print(time_step)
                return time_step

    def SARSA(self):
        self.ALL_INIT()
        print("Start SARSA")
        total_time_step = 0
        episode = 0
        plot_step = []
        while True:
            time_step = self.SARSA_update()

            total_time_step += time_step
            plot_step += [total_time_step]
            if total_time_step > 8000:
                break
            episode +=1
        plt.plot(np.array(plot_step))



    def SARSA_update(self):
        time_step = 0
        state = copy.deepcopy(self.init_state)
        while True:
            if time_step % 1000 == 999:
                print("time step = {:05d}".format(time_step+1))
            time_step+=1
            state_ = copy.deepcopy(state)
            action = self.choose_action(state)

            #### by action
            state_ += self.ACTION[action]

            #### by wind
            if state[0] in [3,4,5,8]:
                state_[1] +=1
            elif state[0] in [6,7]:
                state_[1] +=2

            #### wall
            state_[0] = min(max(state_[0], 0), 9)
            state_[1] = min(max(state_[1], 0), 6)

            action_ = self.choose_action(state_)

            #### Q table update
            self.Q_TABLE_ = copy.deepcopy(self.Q_TABLE)
            self.Q_TABLE_[state[0], state[1], action] += self.ALPHA * (self.STEP_REWARD + self.DISCOUNT_FACTOR * self.Q_TABLE[state_[0], state_[1], action_] - self.Q_TABLE[state[0], state[1], action])

            self.Q_TABLE = copy.deepcopy(self.Q_TABLE_)
            state = copy.deepcopy(state_)

            if state_[0] == 7 and state_[1] == 3:
                print(time_step)
                return time_step


    def choose_action(self, state):
        if random.random() < self.EPS:
            #### exploration
            return random.randint(0,3)          #### randomly choose 0~3 action

        else:
            Q_values = self.Q_TABLE[state[0], state[1]]
            max_val = Q_values.max()
            #### if max_val has multiple max value, choose randomly
            return np.random.choice(np.where(Q_values == max_val)[0])





if __name__ == '__main__':
    PLAY = Windy_Gridworld()
    PLAY.SARSA_lambda()
    PLAY.SARSA()
    plt.legend(["SARSA lambda", "SARSA"])
    plt.savefig("./Windy_Gridworld.png")


