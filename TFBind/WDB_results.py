import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import random
import wandb
from collections import Counter

# Initialize WANDB
wandb.init(project="distributional-gfn-tfbind", config={"learning_rate": 0.001, "epochs": 100}, name="TFBIND", save_code=True)

# Set default device to GPU if available, else CPU
#mx.set_default_device(mx.gpu if mx.gpu.is_available() else mx.cpu)

print(f"Using device: {mx.default_device()}")

class Molecule:
    def __init__(self, sequence):
        self.sequence = sequence

    def mutate(self):
        idx = random.randint(0, len(self.sequence) - 1)
        new_char = random.choice('ACGT')
        mutated_sequence = self.sequence[:idx] + new_char + self.sequence[idx + 1:]
        return Molecule(mutated_sequence)

    def get_representation(self):
        rep = mx.array([[ord(char) for char in self.sequence]], dtype=mx.float32)
        return rep  # Shape: (1, sequence_length)

class TransformerNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.layers = [nn.TransformerEncoderLayer(hidden_size, num_heads) for _ in range(num_layers)]
        self.fc_out = nn.Linear(hidden_size, output_size)

    def __call__(self, x):
        # Ensure input is 3D: (batch_size, sequence_length, input_size)
        if x.ndim == 2:
            x = x.reshape(1, *x.shape)
        
        x = self.embedding(x)
        
        # TransformerEncoderLayer expects (sequence_length, batch_size, hidden_size)
        x = mx.transpose(x, [1, 0, 2])
        
        # Create a mask for the transformer layers
        seq_length = x.shape[0]
        mask = mx.tril(mx.ones((seq_length, seq_length)))
        
        for layer in self.layers:
            x = layer(x, mask)
        
        # Change back to (batch_size, sequence_length, hidden_size)
        x = mx.transpose(x, [1, 0, 2])
        
        x = mx.mean(x, axis=1)
        return self.fc_out(x)


class DistributionalGFlowNet:
    def __init__(self, input_size):
        self.policy_net = TransformerNetwork(input_size, 4)
        self.flow_net = TransformerNetwork(input_size, 2)
        self.optimizer = optim.Adam(learning_rate=wandb.config.learning_rate)
        self.loss_history = []
        self.modes_history = []
        self.diversity_history = []

    def sample_action(self, state):
        logits = self.policy_net(state)
        action_probs = mx.softmax(logits, axis=1)
        action = mx.random.categorical(action_probs)
        return int(action.item())

    def compute_loss(self, trajectories, rewards):
        total_loss = 0
        for trajectory, reward in zip(trajectories, rewards):
            state_vectors = mx.stack(trajectory)
            flow_preds = self.flow_net(state_vectors)
            flow_means = flow_preds[:, 0]
            flow_vars = mx.exp(flow_preds[:, 1])
            distributional_loss = mx.mean((reward - flow_means) ** 2 / flow_vars + mx.log(flow_vars))
            total_loss += distributional_loss
        return total_loss

    def track_metrics(self, trajectories):
        unique_sequences = set([tuple(traj[-1].tolist()[0]) for traj in trajectories])
        num_modes = len(unique_sequences)
        self.modes_history.append(num_modes)

        sequence_counter = Counter([tuple(traj[-1].tolist()[0]) for traj in trajectories])
        probabilities = np.array(list(sequence_counter.values())) / len(trajectories)
        diversity = -np.sum(probabilities * np.log(probabilities))
        self.diversity_history.append(diversity)

        wandb.log({
            "loss": self.loss_history[-1],
            "modes": num_modes,
            "diversity": diversity
        })

    def train(self, environment, epochs=1000):
        for epoch in range(epochs):
            state = environment.reset()
            done = False
            trajectory = []
            rewards = []
            trajectories = []
            while not done:
                action = self.sample_action(state)
                next_state, reward, done = environment.step(action)
                trajectory.append(state)
                rewards.append(reward)
                state = next_state
            
            trajectories.append(trajectory)
            
            loss = self.compute_loss([trajectory], [sum(rewards)])
            grads = mx.grad(self.compute_loss)([trajectory], [sum(rewards)])
            self.optimizer.update(self.policy_net, grads)
            self.optimizer.update(self.flow_net, grads)

            self.loss_history.append(float(loss))
            self.track_metrics(trajectories)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}, Modes: {self.modes_history[-1]}, Diversity: {self.diversity_history[-1]}")

class TFBindEnvironment:
    def __init__(self, initial_sequence):
        self.molecule = Molecule(initial_sequence)

    def reset(self):
        self.molecule = Molecule("ACGTACGT")
        return self.molecule.get_representation()

    def step(self, action):
        self.molecule = self.molecule.mutate()
        reward = self.compute_binding_affinity(self.molecule)
        done = False
        return self.molecule.get_representation(), reward, done

    def compute_binding_affinity(self, molecule):
        return sum([ord(char) for char in molecule.sequence]) % 10

if __name__ == "__main__":
    initial_sequence = "ACGTACGT"
    env = TFBindEnvironment(initial_sequence)
    input_size = len(initial_sequence)
    agent = DistributionalGFlowNet(input_size)
    agent.train(env)