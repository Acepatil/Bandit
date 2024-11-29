import numpy as np
import matplotlib.pyplot as plt

def bandit_nonstat(action, m):
    v = np.random.normal(0, 0.01, 10)
    m = m + v
    value = m[action]
    return value, m

def ten_arm_bandit():
    Q = np.zeros(10)  # Value estimates for each arm
    N = np.zeros(10)  # Number of times each arm has been selected
    R = np.zeros(10000)  # Running average of rewards
    epsilon = 0.1  # Exploration probability
    m = np.zeros(10)  # Mean rewards for each arm
    RR = 0  # Reward variable

    for i in range(10000):
        if np.random.rand() > epsilon:
            A = np.argmax(Q)  # Greedy action
        else:
            A = np.random.randint(0, 10)  # Exploratory action

        RR, m = bandit_nonstat(A, m)  # Get reward and update mean rewards
        N[A] += 1  # Increment count of chosen arm
        Q[A] += (RR - Q[A]) / N[A]  # Update action value estimate

        # Update running average of rewards
        if i == 0:
            R[i] = RR
        else:
            R[i] = ((i - 1) * R[i - 1] + RR) / i

    # Plot the reward progression
    plt.plot(range(1, 10001), R, 'r')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.ylim(0, max(R))
    plt.title('10-Arm Bandit with Non-Stationary Rewards')
    plt.show()

# Call the function to execute the bandit simulation
ten_arm_bandit()
