#### HW#5
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

def MRP():
    DISCOUNT_FACTOR = 1
    Identity = np.identity(n=7)
    probability_mat = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0.5, 0, 0.5, 0, 0, 0, 0],
                            [0, 0.5, 0, 0.5, 0, 0, 0],
                            [0, 0, 0.5, 0, 0.5, 0, 0],
                            [0, 0, 0, 0.5, 0, 0.5, 0],
                            [0, 0, 0, 0, 0.5, 0, 0.5],
                            [0, 0, 0 ,0 ,0, 0, 0]]
                           )

    reward_mat = np.array([[0],
                           [0],
                           [0],
                           [0],
                           [0],
                           [0.5],
                           [0]])

    value_mat = np.matmul(np.linalg.inv(Identity-DISCOUNT_FACTOR*probability_mat),
                          reward_mat)

    print(value_mat)

def TD(ALPHA):
    True_table = np.asarray([(x+1)/6 for x in range(0, 5)])
    DISCOUNT_FACTOR = 1
    EPISODES = 100
    VTABLE = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0])
    RMS_error = []

    for episode in range(EPISODES):
        #### inititalize state
        ## A: 0, B: 1, C: 2, D: 3, E: 4
        state = random.randint(1, 5)
        while True:
            action = int((random.randint(0, 1)-0.5)*2)
            state_ = state + action

            REWARD = 1 if state_ == 6 else 0

            VTABLE[state] = VTABLE[state] + ALPHA * (REWARD + DISCOUNT_FACTOR * VTABLE[state_] - VTABLE[state])

            state = copy.deepcopy(state_)

            if state == 0 or state == 6:
                break
        RMS_error.append(sum((VTABLE[1:-1] - True_table) ** 2))
        if episode == 0 or episode == 1 or episode == 10 or episode == 99:
            plot_table = VTABLE[1:-1]
            plt.plot(plot_table, label=str(episode))
    print(VTABLE)
    plt.plot(np.asarray(True_table), label="TRUE")
    plt.legend(loc = "best")
    plt.title("One step Temporal Difference")
    plt.savefig("./TD{:02d}.png".format(int(ALPHA*100)))
    plt.close()

    return RMS_error


def First_Visit_MC(ALPHA):
    True_table = np.asarray([(x+1)/6 for x in range(0, 5)])
    RMS_error = []
    DISCOUNT_FACTOR = 1
    EPISODES = 100
    VTABLE = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0])
    VTABLE_ = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0])

    for episode in range(EPISODES):
        #### inititalize state
        ## A: 1, B: 2, C: 3, D: 4, E: 5
        COUNTER = [0]*7
        RETURN = []
        STATE = []
        ACTION = []
        state = 3
        #### set action
        while True:
            COUNTER[state] += 1
            action = int((random.randint(0, 1)-0.5)*2)
            ACTION.append(action)

            state_ = state + action
            state = copy.deepcopy(state_)
            if state == 0 or state == 6:
                break


        #### calc return for every state
        state = 3
        for idx, action in enumerate(ACTION):
            RETURN.append(0)
            STATE.append(state)

            state = state + action
            REWARD = 1 if state == 6 else 0


            for idx, R in enumerate(RETURN):
                plus_reward = REWARD if idx == 0 else REWARD * (DISCOUNT_FACTOR ** idx)
                RETURN[idx] = R + plus_reward


        for idx, S in enumerate(STATE):
            VTABLE_[S] = VTABLE[S] + (RETURN[idx] - VTABLE[S]) * ALPHA
            # VTABLE_[S] = VTABLE[S] + (RETURN[idx] - VTABLE[S])/COUNTER[S]
        VTABLE = copy.deepcopy(VTABLE_)
        RMS_error.append(sum((VTABLE[1:-1] - True_table) ** 2))

        if episode == 0 or episode == 1 or episode == 10 or episode == 99:
            plot_table = VTABLE[1:-1]
            plt.plot(plot_table, label=str(episode))

    print(VTABLE)
    plt.plot(np.asarray([(x+1)/6 for x in range(0, 5)]), label="TRUE")
    plt.legend(loc = "best")
    plt.title("First visit Monte-Carlo")
    plt.savefig("MC{:02d}.png".format(int(ALPHA*100)))
    plt.close()

    return RMS_error


if __name__ == '__main__':
    print("MRP result")
    MRP()

    print("TD")
    TD_error1 = TD(0.05)
    TD_error2 = TD(0.1)
    TD_error3 = TD(0.15)


    print("MC")
    MC_error1 = First_Visit_MC(0.01)
    MC_error2 = First_Visit_MC(0.02)
    MC_error3 = First_Visit_MC(0.03)
    MC_error4 = First_Visit_MC(0.04)

    plt.plot(TD_error1, label="TD_0.05")
    plt.plot(TD_error2, label="TD_0.1")
    plt.plot(TD_error3, label="TD_0.15")

    plt.plot(MC_error1, label="MC_0.01")
    plt.plot(MC_error2, label="MC_0.02")
    plt.plot(MC_error3, label="MC_0.03")
    plt.plot(MC_error4, label="MC_0.04")


    plt.legend(loc = "best")
    plt.title("Error")
    plt.savefig("./RMS_Error.png")
    plt.close()