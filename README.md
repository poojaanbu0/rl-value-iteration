# VALUE ITERATION ALGORITHM

## AIM
To find an optimal policy for an agent navigating a grid-world with slippery tiles, aiming to reach a goal state while maximizing expected rewards using value iteration algorithm.

## PROBLEM STATEMENT
The problem involves using the Value Iteration algorithm to find the best strategy for an agent in the Frozen Lake environment. The agent must navigate icy terrain, avoid hazards, and reach the goal while optimizing cumulative rewards in an uncertain environment.

## POLICY ITERATION ALGORITHM
The **Policy Iteration algorithm** is a method used in reinforcement learning to find the optimal policy for a **Markov Decision Process (MDP)**. It iteratively improves a policy by alternating between two key steps:

1. **Policy Evaluation**: In this step, the current policy is evaluated by computing its **value function**, which estimates the expected return (rewards) from each state when following the policy.

2. **Policy Improvement**: Once the value function is computed, the policy is updated by choosing actions that maximize the expected return based on the current value function, effectively improving the policy.

These two steps—evaluation and improvement—are repeated until the policy becomes stable, meaning no further improvement is possible. At this point, the algorithm converges to an optimal policy, ensuring the best possible actions are taken in each state.

### Steps of the Policy Iteration Algorithm:
1. **Initialize** a random policy.
2. **Policy Evaluation**: Compute the value function for the current policy by solving the Bellman equation for each state.
3. **Policy Improvement**: Update the policy by selecting actions that maximize the value function for each state.
4. **Repeat** steps 2 and 3 until the policy no longer changes (convergence).

This algorithm is guaranteed to converge to an optimal policy for an MDP in a finite number of iterations. It is effective but can be computationally expensive for large state spaces.

## VALUE ITERATION FUNCTION
### Name:POOJA A
### Register Number:212222240072
```python
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward+gamma*V[next_state]*(not done))
        if np.max(np.abs(V-np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi= lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return V, pi
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/33004c9c-bf2e-46c8-8f0e-342fb73ff991)

### optimal policy
![image](https://github.com/user-attachments/assets/989b5469-c0dc-411a-a992-bfc8e98527bf)

### optimal value function
![image](https://github.com/user-attachments/assets/175bbf7a-f644-4a03-81d4-6d0a8356b712)

### success rate for the optimal policy
![image](https://github.com/user-attachments/assets/e790d723-7be1-484a-b41d-6f7c8ce30a1b)

## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
