import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from game import CarEnv


# Model
class DQN(nn.Module):
    def __init__(self, in_nodes, h1_nodes, h2_nodes, out_nodes):
        super().__init__()

        self.fc1 = nn.Linear(in_nodes, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.out = nn.Linear(h2_nodes, out_nodes)

        # stored activations
        self.activations = {}

    def forward(self, x):
        x = torch.tensor(x)
        self.activations["input"] = x.detach().cpu() * 2

        x = F.relu(self.fc1(x))
        self.activations["h1"] = x.detach().cpu()

        x = F.relu(self.fc2(x))
        self.activations["h2"] = x.detach().cpu()

        x = self.out(x)
        tensor = x.detach().cpu()
        self.activations["out"] = torch.softmax(tensor, 0) + 0.2

        return x


# Experience Replay
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        # transition (state, action, new_state, reward, terminated)
        self.memory.append(transition)
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

HIDDEN_NODES = 32

class CarDQL:
    # Hyperparameters (as class attributes)
    lr = 0.01  # learning rate
    gamma = 0.95  # discount factor

    network_sync_rate = 500  # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 10000  # size of replay memory
    mini_batch_size = 32  # size of the training data set sampled from replay memory

    # Neural network
    loss_fn = nn.MSELoss()
    optimizer = None  # initialized later as an instance attribute

    def train(self, episodes: int):
        env = CarEnv()
        num_states = env.observation_space_n
        num_actions = env.action_space.n
        env.init_neural_net([num_states, HIDDEN_NODES, HIDDEN_NODES, num_actions])

        epsilon = 1  # exploration rate, 100% random at first
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(num_states, HIDDEN_NODES, HIDDEN_NODES, num_actions)
        target_dqn = DQN(num_states, HIDDEN_NODES, HIDDEN_NODES, num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to another)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Optimizer for the Policy network, cause we train only this one
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.lr)

        # Keep track of total reward evolution
        total_reward_per_episode = np.zeros(episodes)

        # Track number of steps taken. Used for syncing policy => target network.
        sync_step_count = 0
        total_step_count = 0

        # Episodes loop
        for episode in range(episodes):
            print(f"Episode: {episode}, epsilon = {epsilon}")
            env.FPS = env.FPS * 1.05

            # render_this_episode = 0 <= (episode + 1) % 15 < 2
            render_this_episode = True
            if render_this_episode:
                env.init_display()
            else:
                env.close_display()

            state = env.reset()
            done = False
            truncated = False

            steps_at_start = total_step_count

            while (not done and not truncated):
                if render_this_episode:
                    env.episode = episode
                    env.render()

                # Choose action
                if random.random() < epsilon: 
                    # explore, select random actions
                    action = env.action_space.sample()  # 0, 1, 2
                else:
                    # exploit, select best action (max Q-value)
                    with torch.no_grad():
                        # argmax() returns a tensor containing the index of the highest value, 
                        # .item() extracts the raw Python scalar from a single-element tensor
                        action = policy_dqn(state).argmax().item()
                
                # Apply action
                next_state, reward, done, truncated, _ = env.step(action)

                # Increase the total reward for this episode
                total_reward_per_episode[episode] += reward

                # Save memory for Experience Replay
                memory.append((state, action, next_state, reward, done))

                # Train every 4 steps
                if total_step_count % 4 == 0 and len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                env.animate_neural_net(policy_dqn.activations)
                
                # Move to next state
                state = next_state

                total_step_count += 1
                sync_step_count += 1

                # Sync policy network and target network after a certain number of steps
                if sync_step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    sync_step_count = 0  # reset sync step count

            # --- episode done ---
            # Decay epsilon
            # epsilon = max(epsilon - 1/episodes, 0)
            epsilon = max(0.05, epsilon * 0.98) # exponential decay
            steps_this_episode = total_step_count - steps_at_start
            print(f"Steps this episode: {steps_this_episode}")

        # --- all episodes done ---
        print(f"Training done. Ran for {episodes} episodes")
        print(f"Total training steps: {total_step_count}")
        torch.save(policy_dqn.state_dict(), "car_dqn_test.pt")

        # Create a graph
        plt.figure(1)
        plt.plot(total_reward_per_episode)
        plt.savefig('car_dqn_total_reward.png')


    # optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []

        # Experience replaying
        for state, action, next_state, reward, terminated in mini_batch:
            # First, we decide what is the target value we want the Policy network to predict after training
            # It can be either the final reward received after this state & action pair resulted in "done" / terminated

            # There is no next_state (S'), and there are no future actions to take
            if terminated:
                # Agent either reached goal (reward = 100) or hit the wall (reward = -100)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.tensor([reward], dtype=torch.float32)
            else:
                # Otherwise, we choose the max target q-value in the next state from the Target DQN (Bellman equation)
                # The Target DQN serves as a stable anchor point network while we update the Policy network
                with torch.no_grad():
                    next_state_q_values = target_dqn(next_state)
                    target = torch.tensor(
                        [reward + self.gamma * next_state_q_values.max()],
                        dtype=torch.float32
                    )
            
            # Get the current Q-values for the current state, in order to use it for calculating the loss
            current_q = policy_dqn(state)
            current_q_list.append(current_q)

            # Copy the current Q-values to the target values and update the Q-value for the action taken
            # Basically, we increase the expected reward for the action taken in this current state
            # We're backpropagating some of the next state expected reward to the previous state
            target_q = current_q.clone().detach()  # start from current predictions
            target_q[action] = target          # only update the action that was taken
            target_q_list.append(target_q)

        # Compute loss for the whole mini-batch
        # torch.stack() joins [tensor([1,2,3]), tensor([4,5,6])] into tensor([[1,2,3], [4,5,6]])
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def test(self):
        env = CarEnv()
        num_states = env.observation_space_n
        num_actions = env.action_space.n
        env.init_neural_net([num_states, HIDDEN_NODES, HIDDEN_NODES, num_actions])
        env.FPS = 180

        policy_dqn = DQN(num_states, HIDDEN_NODES, HIDDEN_NODES, num_actions)
        policy_dqn.load_state_dict(torch.load("car_dqn_test.pt"))
        policy_dqn.eval()  # switch model to evaluation mode

        while True:
            env.init_display()
            state = env.reset()
            done = False
            truncated = False

            while (not done and not truncated):
                env.episode = None
                env.render()
                # Exploit with the best action in current state
                with torch.no_grad():
                    action = policy_dqn(state).argmax().item()
                
                env.animate_neural_net(policy_dqn.activations)
                
                # Apply action
                next_state, reward, done, truncated, _ = env.step(action, truncate_at=10000)
                
                # Move to next state
                state = next_state


car_dql = CarDQL()
# car_dql.train(episodes=50)
car_dql.test()