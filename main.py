import numpy as np
import torch
from zhan import FJSSP_AM_Env as FJSSPAMEnvironment
from DQNframestruct import QNetwork,ReplayBuffer
import torch.optim as optim


# Define the main function for training the environment
def main():
    # Define the hyperparameters
    num_episodes = 1000
    batch_size = 32
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    hidden_size = 128
    buffer_capacity = 10000

    # Initialize the environment, Q-network, optimizer, and replay buffer
    env = FJSSPAMEnvironment()
    q_network = QNetwork(env.state_size, env.action_size, hidden_size)
    optimizer = optim.Adam(q_network.parameters())
    replay_buffer = ReplayBuffer(buffer_capacity)

    # Train the Q-network
    epsilon = epsilon_start
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        for t in range(env.max_timestep):
            action = select_action(state, q_network, epsilon)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], dtype=torch.float)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            done = torch.tensor([done], dtype=torch.bool)
            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state
            train_dqn(env, q_network, optimizer, replay_buffer, batch_size, gamma)
            if done:
                break
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

# Run the main function
if __name__ == '__main__':
    main()