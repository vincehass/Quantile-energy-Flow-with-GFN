import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

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
        self.loss_history = []
        self.modes_history = []
        self.diversity_history = []

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

    def track_metrics(self, trajectories):
        # Track the number of unique sequences (modes)
        unique_sequences = set([tuple(traj[-1].cpu().numpy()) for traj in trajectories])
        num_modes = len(unique_sequences)
        self.modes_history.append(num_modes)

        # Track diversity (Shannon entropy of sequence distribution)
        sequence_counter = Counter([tuple(traj[-1].cpu().numpy()) for traj in trajectories])
        probabilities = np.array(list(sequence_counter.values())) / len(trajectories)
        diversity = -np.sum(probabilities * np.log(probabilities))
        self.diversity_history.append(diversity)

    def train(self, environment, epochs=1000):
        for epoch in range(epochs):
            state = environment.reset()
            done = False
            trajectory = []
            rewards = []
            trajectories = []  # To keep track of all trajectories in this epoch
            while not done:
                action = self.sample_action(state)
                next_state, reward, done = environment.step(action)
                trajectory.append(state)
                rewards.append(reward)
                state = next_state
            
            trajectories.append(trajectory)
            
            # Compute the loss based on the trajectory and rewards
            loss = self.compute_loss([trajectory], [sum(rewards)])  # Sum of rewards for simplicity
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log the metrics
            self.loss_history.append(loss.item())
            self.track_metrics(trajectories)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}, Modes: {self.modes_history[-1]}, Diversity: {self.diversity_history[-1]}")

    def plot_metrics(self):
        # Plot loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.show()

        # Plot number of modes
        plt.figure(figsize=(10, 5))
        plt.plot(self.modes_history, label="Number of Modes")
        plt.xlabel("Epoch")
        plt.ylabel("Number of Modes")
        plt.title("Number of Modes Over Time")
        plt.legend()
        plt.show()

        # Plot diversity
        plt.figure(figsize=(10, 5))
        plt.plot(self.diversity_history, label="Diversity (Shannon Entropy)")
        plt.xlabel("Epoch")
        plt.ylabel("Diversity")
        plt.title("Diversity Over Time")
        plt.legend()
        plt.show()

# Define the TFBIND environment
class TFBindEnvironment:
    def __init__(self, initial_sequence):
        self.molecule = Molecule(initial_sequence)

    def reset(self):
        # Reset to the initial molecule
        self.molecule = Molecule("ACGTACGT")
        return self.molecule.get_representation()

    def step(self, action):
        # Apply a mutation to the molecule
        self.molecule = self.molecule.mutate()
        reward = self.compute_binding_affinity(self.molecule)
        done = False  # Molecule generation is typically not episodic
        return self.molecule.get_representation(), reward, done

    def compute_binding_affinity(self, molecule):
        # Dummy function to simulate binding affinity evaluation
        return sum([ord(char) for char in molecule.sequence]) % 10  # Arbitrary reward function

# Main execution
if __name__ == "__main__":
    initial_sequence = "ACGTACGT"  # Starting DNA sequence
    env = TFBindEnvironment(initial_sequence)
    input_size = len(initial_sequence)
    agent = DistributionalGFlowNet(input_size)
    agent.train(env)

    # Plot the metrics
    agent.plot_metrics()
