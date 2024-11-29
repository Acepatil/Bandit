import numpy as np
import matplotlib.pyplot as plt

# Bandit functions for binaryBanditA and binaryBanditB
def binary_bandit_a(action):
    """
    Simulates the binaryBanditA.m function.
    :param action: Action to take (1 or 2).
    :return: Reward (1 or 0).
    """
    probabilities = [0.1, 0.2]  # Success probabilities for actions 1 and 2
    return 1 if np.random.rand() < probabilities[action - 1] else 0

def binary_bandit_b(action):
    """
    Simulates the binaryBanditB.m function.
    :param action: Action to take (1 or 2).
    :return: Reward (1 or 0).
    """
    probabilities = [0.8, 0.9]  # Success probabilities for actions 1 and 2
    return 1 if np.random.rand() < probabilities[action - 1] else 0

# Epsilon-greedy algorithm
def epsilon_greedy(bandit_function, epsilon=0.1, iterations=1000):
    """
    Implements the epsilon-greedy algorithm to maximize rewards.
    :param bandit_function: Function representing the bandit (binary_bandit_a or binary_bandit_b).
    :param epsilon: Probability of exploring random actions.
    :param iterations: Number of iterations to run the algorithm.
    :return: Action-value estimates, action counts, average rewards.
    """
    Q = np.zeros(2)  # Estimated value of actions (Q[0] for action 1, Q[1] for action 2)
    N = np.zeros(2)  # Counts of actions taken
    avg_rewards = []

    for i in range(1, iterations + 1):
        # Choose action
        if np.random.rand() > epsilon:
            action = np.argmax(Q) + 1  # Exploit (1-based indexing for actions)
        else:
            action = np.random.choice([1, 2])  # Explore

        # Get reward from the chosen action
        reward = bandit_function(action)

        # Update action-value estimates
        N[action - 1] += 1
        Q[action - 1] += (reward - Q[action - 1]) / N[action - 1]

        # Calculate average reward
        if i == 1:
            avg_rewards.append(reward)
        else:
            avg_rewards.append(((i - 1) * avg_rewards[-1] + reward) / i)

    return Q, N, avg_rewards

# Run epsilon-greedy on binaryBanditA and binaryBanditB
Q_values_a, action_counts_a, average_rewards_a = epsilon_greedy(binary_bandit_a, epsilon=0.1, iterations=1000)
Q_values_b, action_counts_b, average_rewards_b = epsilon_greedy(binary_bandit_b, epsilon=0.1, iterations=1000)

# Plotting average rewards for both binaryBanditA and binaryBanditB
plt.figure(figsize=(12, 6))

# Graph for binaryBanditA
plt.subplot(1, 2, 1)
plt.plot(average_rewards_a, color="blue")
plt.title("Average Rewards Over Iterations (binaryBanditA)")
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.ylim(0, 1)
plt.grid(True)

# Graph for binaryBanditB
plt.subplot(1, 2, 2)
plt.plot(average_rewards_b, color="red")
plt.title("Average Rewards Over Iterations (binaryBanditB)")
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.show()

# Print results for both bandits
print("BinaryBanditA:")
print(f"Q-values: {Q_values_a}")
print(f"Action counts: {action_counts_a}")

print("\nBinaryBanditB:")
print(f"Q-values: {Q_values_b}")
print(f"Action counts: {action_counts_b}")
