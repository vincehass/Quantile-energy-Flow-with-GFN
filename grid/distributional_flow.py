import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Let's set up a simple grid environment
# The grid is represented as a 2D array, where each position is a state
GRID_SIZE = 5  # This represents a 5x5 grid
N_ACTIONS = 4  # Actions: up, down, left, right

# Define the environment
class GridEnvironment:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.state = (0, 0)  # Start at the top-left corner of the grid

    def reset(self):
        # Reset the environment to the initial state
        self.state = (0, 0)
        return self.state

    def step(self, action):
        # Move in the grid based on the action
        x, y = self.state
        if action == 0 and y < self.grid_size - 1:  # Move right
            y += 1
        elif action == 1 and y > 0:  # Move left
            y -= 1
        elif action == 2 and x < self.grid_size - 1:  # Move down
            x += 1
        elif action == 3 and x > 0:  # Move up
            x -= 1
        self.state = (x, y)
        done = self.state == (self.grid_size - 1, self.grid_size - 1)  # Terminal state at bottom-right
        reward = 1.0 if done else 0.0  # Reward only at the terminal state
        return self.state, reward, done

# Define a simple neural network for policy and flow prediction
class SimpleNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Output raw values for policy or flow

# Define the distributional GFlowNet agent
class DistributionalGFlowNet:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.policy_net = SimpleNetwork(grid_size * grid_size, N_ACTIONS)  # Predicts policy (action probabilities)
        self.flow_net = SimpleNetwork(grid_size * grid_size, 2)  # Predicts flow distribution (mean and variance)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.flow_net.parameters()), lr=0.001)
    
    def get_state_representation(self, state):
        # Convert the 2D grid position (x, y) into a one-hot vector representation
        state_vector = torch.zeros(self.grid_size * self.grid_size)
        x, y = state
        state_vector[x * self.grid_size + y] = 1.0
        return state_vector

    def sample_action(self, state):
        # Use the policy network to sample an action based on the current state
        state_vector = self.get_state_representation(state)
        logits = self.policy_net(state_vector)
        action_probs = torch.softmax(logits, dim=0)  # Convert to probabilities
        action = torch.multinomial(action_probs, 1).item()  # Sample an action based on the probabilities
        return action

    def compute_loss(self, trajectories, rewards):
        # Placeholder for trajectory balance loss computation
        # Here we will compute the loss based on the flow distribution and rewards
        total_loss = 0
        for trajectory, reward in zip(trajectories, rewards):
            # Convert states in the trajectory to vector form
            state_vectors = torch.stack([self.get_state_representation(s) for s in trajectory])
            
            # Predict flow distribution (mean and variance) for each state
            flow_preds = self.flow_net(state_vectors)
            flow_means = flow_preds[:, 0]
            flow_vars = torch.exp(flow_preds[:, 1])  # Variance should be positive, so we apply exp
            
            # Compute loss based on distributional predictions (simplified for clarity)
            distributional_loss = torch.mean((reward - flow_means) ** 2 / flow_vars + torch.log(flow_vars))
            total_loss += distributional_loss
        return total_loss

    def train(self, environment, epochs=1000):
        for epoch in range(epochs):
            state = environment.reset()
            done = False
            trajectory = []
            rewards = []
            while not done:
                action = self.sample_action(state)
                next_state, reward, done = environment.step(action)
                trajectory.append(state)
                rewards.append(reward)
                state = next_state
            
            # Compute the loss based on the trajectory and rewards
            loss = self.compute_loss([trajectory], [sum(rewards)])  # Sum of rewards for simplicity
            
            # Perform backpropagation and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

# Main execution
if __name__ == "__main__":
    env = GridEnvironment(GRID_SIZE)
    agent = DistributionalGFlowNet(GRID_SIZE)
    agent.train(env)
