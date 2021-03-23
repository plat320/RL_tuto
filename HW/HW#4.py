#### HW#4
import numpy as np
import copy

if __name__ == '__main__':
    #### equation
    #### v^(k+1)=R^pi + \gamma p^pi v_k
    DISCOUNT_FACTOR = 1
    ITER = 1000
    #### table init
    TABLE = np.zeros((4,4))
    TABLE_ = np.zeros((4,4))
    ACTION = ([1,0], [0,1], [-1,0], [0,-1])

    for iter in range(ITER):
        for col in range(4):
            for row in range(4):
                #### terminal state
                if (col == 0 and row == 0) or (col == 3 and row == 3):
                    TABLE_[col, row] = 0
                    continue
                v_ = -1
                for action in ACTION:
                    s_ = np.array([col, row]) + action
                    s_ = np.clip(s_, 0, 3)
                    v_ += TABLE[s_[0], s_[1]]/4

                TABLE_[col, row] = v_
        TABLE = copy.deepcopy(TABLE_)
        if iter == 0 or iter == 1 or iter == 2 or iter == 9 or iter == 99 or iter == 999:
            print("\nITER = {}".format(iter + 1))
            print(TABLE)





