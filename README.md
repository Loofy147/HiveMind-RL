# Autonomous Swarm Intelligence using Deep R-Learning and Graph Attention Networks

This project implements a sophisticated multi-agent system that leverages deep reinforcement learning and graph neural networks to simulate autonomous swarm intelligence. The agents, categorized as "Scouts" and "Miners," operate in a dynamic environment with the goal of collecting minerals while avoiding hazards.

## Key Features

- **Deep Reinforcement Learning:** The agents are trained using a deep R-learning algorithm, which allows them to learn complex behaviors and adapt to their environment.
- **Graph Attention Networks:** A graph attention network is used to enable communication and coordination between agents, allowing them to work together as a cohesive swarm.
- **Dynamic Environment:** The simulation features a dynamic environment with moving hazards, resource-rich asteroids, and a mobile base station, providing a challenging and unpredictable setting for the agents.
- **Modular Architecture:** The codebase is organized into distinct modules for the environment, neural network, and training logic, making it easy to extend and modify.

## How to Run

To run the simulation, you will need to have Python 3 and the following libraries installed:

- PyTorch
- NumPy
- Matplotlib

### Training

To train the agents, run the following command:

```bash
python legion_swarm.py --mode train
```

This will initiate the training process and save the trained model to the `outputs` directory.

### Playback

To watch a playback of the trained agents, use the following command:

```bash
python legion_swarm.py --mode play
```

This will load the trained model and generate a GIF of the swarm's behavior in the `outputs` directory.
