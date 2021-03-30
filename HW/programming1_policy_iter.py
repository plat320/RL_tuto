#### programming 1
import numpy as np
from itertools import product
import math
import copy
import time


def poisson_prob(n, lamda):
    return (lamda**n) * math.exp(-lamda) / math.factorial(n)

def calc_value(req1, ret1, req2, ret2, mid_col, mid_row, RENTAL_CREDIT, VTABLE,
               POI_PROB=None, first_POI_PROB=None, second_POI_PROB=None):

    prob = first_POI_PROB[req1, ret1] * second_POI_PROB[req2, ret2]

    req1 = min(mid_col, req1)
    req2 = min(mid_row, req2)

    col_ = max(min(mid_col + ret1 - req1, 20), 0)
    row_ = max(min(mid_row + ret2 - req2, 20), 0)

    reward = RENTAL_CREDIT*(req1+req2)

    return prob*reward, prob*VTABLE[int(col_), int(row_)]



def calc_row_col_rewards(col, row, action, MOVING_COST, RENTAL_CREDIT, VTABLE,
                         POI_PROB=None, first_POI_PROB=None, second_POI_PROB=None):
    total_credit_reward = 0
    total_discount_reward = 0

    action_reward = abs(action) * MOVING_COST

    #### status after greedy action
    mid_col, mid_row = col - action, row + action

    ## for hoxy exception
    assert mid_col <= 20, "column value more than 20"
    assert mid_row <= 20, "row value more than 20"
    assert mid_col >= 0, "column value less than 0"
    assert mid_row >= 0, "row value less than 0"

    #### Poisson process

    for req1 in range(11):
        for ret1 in range(11):
            for req2 in range(11):
                for ret2 in range(11):
                    credit_reward, discount_reward = calc_value(req1, ret1, req2, ret2, mid_col, mid_row, RENTAL_CREDIT, VTABLE,
                                                                first_POI_PROB=first_POI_PROB, second_POI_PROB=second_POI_PROB)
                    total_credit_reward += credit_reward
                    total_discount_reward += discount_reward

    return action_reward, total_credit_reward, total_discount_reward





if __name__ == '__main__':
    DISCOUNT_FACTOR = 0.9
    ITER = 5
    CONVERGE = 100
    MOVING_COST = -2
    RENTAL_CREDIT = 10
    #### table init
    VTABLE = np.zeros((21,21))
    VTABLE_ = np.zeros((21,21))
    ACTION_TABLE = np.zeros((21,21))
    ACTION_TABLE_ = np.zeros((21,21))
    ACTION = np.arange(-5, 6, 1)        #### first(col) to second(row)

    #### Poisson probability for each location
    POI_PROB4 = np.array([[poisson_prob(y, 4)] for y in range(11)])
    POI_PROB2 = np.array([[poisson_prob(y, 2) for y in range(11)]])
    second_POI_PROB = np.matmul(POI_PROB4, POI_PROB2)

    POI_PROB3 = np.array([[poisson_prob(y, 3) for y in range(11)]])
    first_POI_PROB = np.matmul(POI_PROB3.T, POI_PROB3)

    #### Start iteration
    for iter in range(ITER):
        POLICY_UPDATE_FLAG = False
        while True:
            stime = time.time()
            print("\nITER = {}".format(iter + 1))
            print("update VALUE TABLE")
            for col in range(21):
                for row in range(21):
                    #### greedy action
                    action = ACTION_TABLE[col, row]
                    action_reward, total_credit_reward, total_discount_reward = calc_row_col_rewards(col, row, action, MOVING_COST, RENTAL_CREDIT, VTABLE,
                                                                                                     first_POI_PROB=first_POI_PROB, second_POI_PROB=second_POI_PROB)
                    VTABLE_[col, row] = action_reward + total_credit_reward + DISCOUNT_FACTOR*total_discount_reward

            #### update value table
            print("update value time: {:.4f}".format(time.time()-stime))
            if np.sum(np.abs(VTABLE - VTABLE_)) < CONVERGE:
                POLICY_UPDATE_FLAG = True
            VTABLE = copy.deepcopy(VTABLE_)
            print(VTABLE.astype(int))

            if POLICY_UPDATE_FLAG:
                break



        stime = time.time()
        #### update action table
        print("update POLICY TABLE")
        for col in range(21):
            for row in range(21):
                action_quality_value = np.zeros((11))
                for idx, action in enumerate(ACTION):
                    #### action exception
                    if col - action < 0 or col - action > 20:
                        continue
                    if row + action < 0 or row + action > 20:
                        continue

                    action_reward, total_credit_reward, total_discount_reward = calc_row_col_rewards(col, row, action, MOVING_COST, RENTAL_CREDIT, VTABLE,
                                                                                                     first_POI_PROB=first_POI_PROB, second_POI_PROB=second_POI_PROB)
                    action_quality_value[idx] = action_reward + total_credit_reward + total_discount_reward * DISCOUNT_FACTOR

                ACTION_TABLE_[col, row] = ACTION[np.argmax(action_quality_value)]
        ACTION_TABLE = copy.deepcopy(ACTION_TABLE_)
        ACTION_TABLE_ = np.zeros((21,21))
        print(ACTION_TABLE.astype(int))
        print("update policy time: {:.4f}".format(time.time()-stime))

