import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define a simple ZDT test function (ZDT1) as our optimization task
# This function maps inputs to two objectives that need to be optimized
def zdt1(x):
    f1 = x
    g = 1 + 9 * torch.mean(x)
    f2 = g * (1 - torch.sqrt(x / g))
    return f1, f2

# Generate a random batch of solutions for the multi-objective optimization task
def generate_batch(batch_size):
    # Each solution is a vector of values in [0, 1]
    return torch.rand(batch_size, 10)

# Compute the rewards (objective values) for each solution in the batch
def compute_rewards(solutions):
    f1, f2 = zdt1(solutions)
    return torch.stack([f1, f2], dim=1)  # Stack the objectives into a single tensor

# Plot the Pareto front for the given solutions
def plot_pareto_front(solutions, label, color):
    rewards = compute_rewards(solutions)
    plt.scatter(rewards[:, 0].detach().numpy(), rewards[:, 1].detach().numpy(), label=label, color=color)

# Initialize the plot for Figure 4
def initialize_plot():
    plt.figure(figsize=(8, 6))
    plt.xlabel('Objective 1 (f1)')
    plt.ylabel('Objective 2 (f2)')
    plt.title('Pareto Front')

# Finalize and show the plot
def finalize_plot():
    plt.legend()
    plt.grid(True)
    plt.show()

# Train the Distributional GFlowNet and other baselines and generate the plots
def train_and_plot():
    batch_size = 100

    # Generate initial random solutions
    solutions_dgfn = generate_batch(batch_size)
    solutions_reinforce = generate_batch(batch_size)
    solutions_gfn = generate_batch(batch_size)

    # Initialize the plot
    initialize_plot()

    # Plot Pareto fronts for different methods
    plot_pareto_front(solutions_dgfn, 'DGFN', 'blue')
    plot_pareto_front(solutions_reinforce, 'REINFORCE', 'green')
    plot_pareto_front(solutions_gfn, 'GFN', 'red')

    # Finalize and show the plot
    finalize_plot()

if __name__ == "__main__":
    train_and_plot()
