#### programming 1
import numpy as np
from itertools import product
import math
import copy
import time


class Jack_Car_RENTAL():
    def __init__(self):
        super(Jack_Car_RENTAL, self).__init__()
        self.DISCOUNT_FACTOR = 0.9
        self.MOVING_COST = -2
        self.RENTAL_CREDIT = 10

        self.VTABLE = np.zeros((21, 21))
        self.VTABLE_ = np.zeros((21, 21))
        self.ACTION_TABLE = np.zeros((21, 21))
        self.ACTION_TABLE_ = np.zeros((21, 21))
        self.ACTION = np.arange(-5, 6, 1)  #### first(col) to second(row)

        #### Poisson probability for each location
        self.POI_PROB2 = np.array([self.poisson_prob(y, 2) for y in range(11)])
        self.POI_PROB3 = np.array([self.poisson_prob(y, 3) for y in range(11)])
        self.POI_PROB4 = np.array([self.poisson_prob(y, 4) for y in range(11)])

        self.states = []
        for i in range(21):
            for j in range(21):
                self.states.append([i, j])
        self.customer = []
        for i in range(11):
            for j in range(11):
                self.customer.append([i, j])

    def poisson_prob(self, n, lamda):
        return (lamda ** n) * math.exp(-lamda) / math.factorial(n)

    def calc_value(self, req1, ret1, req2, ret2, mid_col, mid_row):
        prob = self.POI_PROB3[ret1] * self.POI_PROB2[ret2]

        col_ = max(min(mid_col + ret1 - req1, 20), 0)
        row_ = max(min(mid_row + ret2 - req2, 20), 0)

        return prob, self.VTABLE[int(col_), int(row_)]

    def calc_credit_reward(self, req1, req2, mid_col, mid_row):
        prob = self.POI_PROB3[req1] * self.POI_PROB4[req2]

        req1 = min(mid_col, req1)
        req2 = min(mid_row, req2)

        return prob, self.RENTAL_CREDIT * (req1+req2)


    def calc_rewards(self, col, row, action):
        total_credit_reward = 0
        total_discount_reward = 0

        action_reward = abs(action) * self.MOVING_COST

        #### status after greedy action
        mid_col, mid_row = col - action, row + action

        #### Poisson process
        for req1, req2 in self.customer:
            req_prob, credit_reward = self.calc_credit_reward(req1, req2, mid_col, mid_row)
            for ret1, ret2 in self.customer:
                ret_prob, discount_reward = self.calc_value(req1, ret1, req2, ret2, mid_col, mid_row)
                total_discount_reward += req_prob * ret_prob * discount_reward
                total_credit_reward += req_prob * ret_prob * credit_reward

        Final_value = action_reward + total_credit_reward + self.DISCOUNT_FACTOR * total_discount_reward
        return Final_value

    def Value_Update(self):
        stime = time.time()
        for col, row in self.states:
            action = self.ACTION_TABLE[col, row]
            self.VTABLE_[col, row] = self.calc_rewards(col, row, action)

        self.VTABLE = copy.deepcopy(self.VTABLE_)
        print(self.VTABLE.astype(int))
        print("update value time: {:.4f}".format(time.time()-stime))

    def Policy_Update(self):
        stime = time.time()
        for col, row in self.states:
            action_quality_value = np.zeros((11))

            for idx, action in enumerate(self.ACTION):
                #### action exception
                if (col - action < 0 or col - action > 20) or (row + action < 0 or row + action > 20):
                    continue

                action_quality_value[idx] = self.calc_rewards(col, row, action)


            self.ACTION_TABLE_[col, row] = self.ACTION[np.argmax(action_quality_value)]
        self.ACTION_TABLE = copy.deepcopy(self.ACTION_TABLE_)
        self.ACTION_TABLE_ = np.zeros((21,21))
        print(self.ACTION_TABLE.astype(int))
        print("update policy time: {:.4f}".format(time.time() - stime))



if __name__ == '__main__':
    Prob_Solver = Jack_Car_RENTAL()
    ITER = 15
    #### Start iteration
    for iter in range(ITER):
        print("\nITER = {}".format(iter + 1))
        Prob_Solver.Value_Update()

        Prob_Solver.Policy_Update()
