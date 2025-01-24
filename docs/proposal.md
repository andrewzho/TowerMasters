---
layout: default
title: Proposal
---

## Summary of the Project
The **Collaborative Obstacle Tower Solver Using Multi-Agent Reinforcement Learning** project aims to develop a multi-agent reinforcement learning (MARL) system to solve procedurally generated puzzles in the Obstacle Tower environment. The environment challenges agents to navigate increasingly complex locomotion puzzles requiring coordination, planning, and adaptation. 

The system will take inputs such as the visual representation of the tower levels, agent states, and task-specific constraints. Outputs will include coordinated actions taken by multiple agents to progress through the tower efficiently. This project focuses on enabling agents to collaborate, generalize across unseen levels, and address challenges in multi-agent coordination and learning. Applications include training cooperative AI for dynamic problem-solving and generalizing solutions to real-world tasks requiring teamwork.

## AI/ML Algorithms
The project will leverage the following algorithms:
- **Proximal Policy Optimization (PPO)**: For training agents in continuous action spaces.
- **Advantage Actor-Critic (A2C)**: To enable stable and efficient multi-agent learning.
- **Experience Replay Buffers**: To improve learning stability by reusing past experiences.

## Evaluation Plan
### Quantitative Evaluation
1. **Success Rate**: Percentage of tower levels successfully completed by the agents.
2. **Efficiency Metrics**: Average steps required to complete each level.
3. **Reward Accumulation**: Total rewards earned by the agents per episode.

### Qualitative Evaluation
1. **Visualization**: Highlight how agents collaborate to solve puzzles and navigate obstacles.
2. **Emergent Behavior**: Observe the strategies and coordination dynamics developed by agents.
3. **Generalization**: Evaluate performance on unseen, procedurally generated levels.

### Moonshot Case
The moonshot goal is to develop agents capable of solving all levels of the Obstacle Tower environment while demonstrating highly efficient and human-like problem-solving capabilities.

## Meet the Instructor
We plan to meet with the instructor no later than **Week 5** to discuss progress and gather feedback. The suggested earliest meeting date is **[Insert Date]**, ensuring that all teammates can attend.

## AI Tool Usage
We will use the following tools and frameworks:
- **Stable-Baselines3 or RLlib**: For implementing and training the reinforcement learning agents.
- **Obstacle Tower API**: To simulate the environment and evaluate agent performance.
- **Visualization Tools**: For analyzing agent behavior and collaboration strategies.

All AI tools will be documented, including how they were used and their contributions to the project. We will critically assess the quality of AI-generated solutions and improve the system iteratively.
