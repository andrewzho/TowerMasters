---
layout: default
title: Proposal
---
## Summary of the Project

The **Obstacle Tower Challenge: RL for Generalized Problem-Solving** project aims to develop an autonomous RL agent capable of solving procedurally generated puzzles and navigating through increasingly complex environments. Unlike traditional RL agents that can overfit to static environments, this agent will employ **meta-learning, continual learning, and hierarchical RL** techniques to generalize across different levels.

The system will take inputs such as the agent’s current state, level structure, detected obstacles, and available actions. Outputs will include movement decisions, puzzle-solving strategies, and adaptive exploration techniques. This project focuses on enabling an RL agent to develop generalized problem-solving capabilities, making it a novel application of reinforcement learning in procedurally generated environments.

## AI/ML Algorithms

We anticipate using **meta-reinforcement learning** techniques such as **Model-Agnostic Meta-Learning (MAML)** and **Probabilistic Embeddings for Actor-Learner Architectures (PEARL)** to help the agent learn across different environments. Additionally, **Proximal Policy Optimization (PPO)** and **Advantage Actor-Critic (A2C)** will be used to optimize policy learning. Continual learning techniques will ensure that the agent retains useful knowledge across different levels without catastrophic forgetting.

## Evaluation Plan

### Quantitative Evaluation
1. **Level Completion Rate**: Percentage of procedurally generated levels successfully completed.
2. **Time Efficiency**: Average time taken to complete each level.
3. **Exploration Strategy**: Measure how efficiently the agent navigates unfamiliar layouts.
4. **Generalization Performance**: Success rate of the agent on unseen levels compared to previously encountered ones.

### Qualitative Evaluation
1. **Visualization**: Analyze how the agent adapts to new levels in real-time.
2. **Emergent Behavior**: Observe whether the agent develops efficient navigation and problem-solving techniques beyond brute force methods.
3. **Adaptability**: Evaluate how well the agent adjusts its strategy when faced with novel puzzles or obstacles.

### Moonshot Case
The ultimate goal is to develop an agent that **can continuously adapt to new environments without retraining**, autonomously improving its performance as it encounters more complex levels in the Obstacle Tower Challenge.

## Meet the Instructor
Plan to meet the instructor by **Week 5** to discuss project progress and refinements.

## AI Tool Usage
We will use the following tools:
- **Obstacle Tower Environment**: A procedurally generated RL environment for testing generalization capabilities.
- **Stable-Baselines3 or RLlib**: For implementing and training the RL agent.
- **Custom Reward Shaping**: To encourage efficient problem-solving and generalization.

All AI tools used will be documented, including their contributions to the project.

---
