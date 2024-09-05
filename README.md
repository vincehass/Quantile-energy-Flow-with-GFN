# Quantile-energy-Flow-with-GFN

Stochastic Rewards as Distributional Flows

This repository `Quantile-Energy-Flow-with-GFlowNets` is focused on the implementation of Distributional Generative Flow Networks (GFlowNets) for local search. Specifically, the `distributional_flow.py` file located in the `grid` directory contains the implementation of a distributional version of GFlowNets designed to work on grid-based environments. This type of environment serves as a simplified testbed for evaluating the performance of GFlowNets under controlled conditions.

### Overview of `distributional_flow.py`

This file implements the core components of Distributional GFlowNets in a grid-based environment. Distributional GFlowNets extend the GFlowNet framework by modeling the flow function as a distribution rather than a deterministic or point-based value. This allows the network to generate diverse outputs and handle uncertainty in the environment, making it more robust in scenarios with stochastic rewards.

#### 1. **Imports and Dependencies**

- The file imports standard Python libraries (`torch`, `torch.nn`, `torch.optim`, `numpy`) and any necessary components for neural network construction and optimization.
- Utility functions and environment definitions specific to grid-based tasks are also imported from the repository.

#### 2. **Environment Setup**

- The script operates in a grid world, where each state corresponds to a position in the grid. The agent, controlled by the GFlowNet, moves through the grid to reach terminal states.
- The environment may also define reward functions that the agent seeks to maximize by generating trajectories that lead to high-reward terminal states.

#### 3. **Distributional Flow Model**

- At the heart of the `distributional_flow.py` file is the implementation of a GFlowNet where the flow function is modeled as a distribution. This contrasts with classical GFlowNets, where the flow is typically represented as scalar values.
- **Key Idea**: Instead of predicting a single flow value for each state or transition, the network predicts a distribution (e.g., a Gaussian distribution or a mixture of distributions). This distribution captures the uncertainty or variability in the flow across different trajectories.

The model is likely implemented as a neural network, with a forward pass that takes in the current state and outputs the parameters of the distribution representing the flow at that state.

- **Distributional Modeling**: The predicted distributions can be parameterized using a mean and variance (in the case of Gaussian distributions) or by using other distributional families. The idea is to have the network learn not only the expected flow but also the variability around that expectation.

#### 4. **Training Objectives**

- The training process involves optimizing the flow distributions to satisfy the flow constraints defined by GFlowNets. Specifically, the total incoming flow into any state should equal the total outgoing flow from that state, adjusted by any rewards received at terminal states.
- **Trajectory Balance Loss**: The script likely includes an objective function based on trajectory balance, a core principle in GFlowNets that ensures consistency across different trajectories leading to the same outcome.
- **Distributional Loss**: For the distributional version, the loss function might be extended to account for the discrepancy between the predicted distributions and the observed returns from different trajectories.

The training objective could involve minimizing a loss function that penalizes deviations from the desired flow balance while encouraging the distributions to accurately model the flow at each state.

#### 5. **Policy and Flow Parameterization**

- The agent's policy is represented by a neural network that defines the transition probabilities between states in the grid. The flow is parameterized by another neural network (or the same network) that models the distributional flow function.
- The agent learns a policy that not only maximizes the expected reward but also matches the distribution of flows as prescribed by the GFlowNet framework.

#### 6. **Sampling and Action Selection**

- During training and evaluation, the agent generates trajectories by sampling actions according to the policy network's output. The sampled trajectories are used to compute the flow distributions and update the network parameters through backpropagation.
- **Exploration vs. Exploitation**: The use of distributional flows can help balance exploration and exploitation by enabling the agent to explore a diverse set of trajectories while still being guided towards high-reward regions of the grid.

#### 7. **Optimization and Updates**

- The script includes the optimization loop where the neural networks are trained. This typically involves iterating through episodes, collecting trajectories, calculating the trajectory balance loss, and updating the network weights using an optimizer like Adam.
- **Gradient Descent**: The backpropagation step computes gradients of the loss with respect to the network parameters and applies gradient descent to minimize the loss function.

The use of distributional loss functions adds complexity to this step, as gradients need to be computed not only for the expected value of the flow but also for the parameters of the distribution (e.g., mean and variance).

#### 8. **Evaluation**

- After training, the model is evaluated by generating trajectories and assessing the quality of the generated outputs. For a grid environment, this could involve measuring the distribution of terminal states reached by the agent and comparing it to the target distribution.
- The evaluation process provides insights into how well the distributional GFlowNet has learned to balance exploration and exploitation and how effectively it handles uncertainty in the environment.

### Key Concepts:

- **Distributional GFlowNets**: A generalization of GFlowNets where flows are modeled as distributions rather than point estimates. This enables the model to handle uncertainty and generate diverse outputs.
- **Grid Environment**: A simple testbed used to evaluate the performance of the model. The grid represents a structured environment where the agent moves between discrete states to reach terminal states.
- **Trajectory Balance**: A core principle in GFlowNets ensuring that the flow into and out of states is balanced, which leads to consistent trajectory generation.
- **Neural Networks**: Used to parameterize the policy and flow distributions. The networks are trained using gradient-based optimization to satisfy the GFlowNet objectives.

In summary, `distributional_flow.py` implements a distributional extension of GFlowNets for grid-based environments. The script models the flow function as a distribution, introduces a corresponding loss function, and optimizes the model to generate trajectories that respect the GFlowNet flow balance constraints. By operating in a grid environment, this implementation provides a controlled setting to test and validate the performance of distributional GFlowNets.
