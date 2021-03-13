#### HW#2
import numpy as np

if __name__ == '__main__':
    DISCOUNT_FACTOR = 1
    Identity = np.identity(n=5)
    probability_mat = np.array([[0.5, 0.5, 0, 0, 0],
                            [0.5, 0, 0.5, 0, 0],
                            [0, 0, 0, 0.5, 0.5],
                            [0, 0.1, 0.2, 0.2, 0.5],
                            [0, 0, 0, 0, 0]]
                           )

    reward_mat = np.array([[-0.5],
                           [-1.5],
                           [-1],
                           [5.5],
                           [0]])

    value_mat = np.matmul(np.linalg.inv(Identity-DISCOUNT_FACTOR*probability_mat),
                          reward_mat)

    print(value_mat)