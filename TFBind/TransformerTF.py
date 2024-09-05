import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

'''This setup incorporates Transformer models into the policy and flow networks, allowing the model to better handle sequential 
data and capture complex dependencies. Adjust the hyperparameters like num_heads and num_layers based on your 
specific requirements and available computational resources.
'''



# Check for GPU support (MPS or CUDA) and set the device accordingly
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define a simple representation of molecules
class Molecule:
    def __init__(self, sequence):
        self.sequence = sequence  # Sequence of characters representing the molecule

    def mutate(self):
        # Apply a random mutation to the sequence
        idx = random.randint(0, len(self.sequence) - 1)
        new_char = random.choice('ACGT')  # Assume the molecule is a DNA sequence with ACGT bases
        mutated_sequence = self.sequence[:idx] + new_char + self.sequence[idx + 1:]
        return Molecule(mutated_sequence)

    def get_representation(self):
        # Convert the sequence to a numerical representation
        return torch.tensor([ord(char) for char in self.sequence], dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension

# Define the Transformer-based neural network for policy and flow prediction
class TransformerNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_heads=4, num_layers=2):
        super(TransformerNetwork, self).__init__()
        self.embedding = nn.Linear(input_size, 128)
        self.transformer = nn.Transformer(
            d_model=128,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(128, output_size)

    def forward(self, x):
        # x should have shape (seq_len, batch, input_size)
        x = self.embedding(x)
        x = self.transformer(x, x)  # Use same input for source and target for simplicity
        x = x.mean(dim=0)  # Aggregate sequence information
        return self.fc_out(x)

# Define the Distributional GFlowNet for the TFBIND task with Transformers
class DistributionalGFlowNet:
    def __init__(self, input_size):
        self.policy_net = TransformerNetwork(input_size, 4).to(device)
        self.flow_net = TransformerNetwork(input_size, 2).to(device)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.flow_net.parameters()), lr=0.001)

    def sample_action(self, state):
        # Use the policy network to sample an action (mutation) based on the current molecule representation
        logits = self.policy_net(state)
        action_probs = torch.softmax(logits, dim=1)  # Convert to probabilities
        action = torch.multinomial(action_probs.squeeze(), 1).item()  # Sample an action based on the probabilities
        return action

    def compute_loss(self, trajectories, rewards):
        total_loss = 0
        for trajectory, reward in zip(trajectories, rewards):
            state_vectors = torch.stack(trajectory).to(device)
            flow_preds = self.flow_net(state_vectors)
            flow_means = flow_preds[:, 0]
            flow_vars = torch.exp(flow_preds[:, 1])  # Variance should be positive, so we apply exp
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
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

# Main execution
if __name__ == "__main__":
    initial_sequence = "ACGTACGT"  # Starting DNA sequence
    env = TFBindEnvironment(initial_sequence)
    input_size = len(initial_sequence)
    agent = DistributionalGFlowNet(input_size)
    agent.train(env)
