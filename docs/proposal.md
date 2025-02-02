---
layout: default
title: Proposal
---

## Summary of the Project
The **Self-Sufficient Mining Bot in Minecraft Using Reinforcement Learning** project aims to develop an autonomous RL agent capable of mining resources, crafting tools, storing materials, and gathering food to sustain itself. Unlike traditional scripted bots, this agent will learn optimal strategies for long-term survival and resource management, allowing it to function independently within the Minecraft world. 

The system will take inputs such as the agent’s current inventory, terrain conditions, nearby resource availability, and health status. Outputs will include movement actions, mining decisions, crafting selections, and food consumption strategies. This project focuses on enabling an RL agent to become self-sufficient by managing both short-term mining efficiency and long-term survival, making it a unique approach to AI-driven automation in open-world environments.

## AI/ML Algorithms
We anticipate using **reinforcement learning with neural function approximators**, specifically leveraging **Proximal Policy Optimization (PPO)** and **Advantage Actor-Critic (A2C)** methods. These algorithms will be model-free and primarily on-policy, allowing the agent to adapt to its environment and learn optimal mining, crafting, and resource management strategies.

## Evaluation Plan
### Quantitative Evaluation
1. **Mining Efficiency**: Number of valuable resources (diamonds, iron, coal) collected per hour.
2. **Tool Utilization**: Number of crafted tools vs. their effective usage rate.
3. **Survival Rate**: The agent’s ability to maintain health over extended time periods.
4. **Autonomy Duration**: How long the agent can operate without external input.

### Qualitative Evaluation
1. **Visualization**: Highlight how the agent mines, navigates, and crafts efficiently.
2. **Emergent Behavior**: Observe whether the agent develops efficient mining techniques, such as tunnel placement or staircasing.
3. **Self-Sufficiency**: Evaluate how well the agent balances mining, crafting, and food gathering over extended periods.

### Moonshot Case
The ultimate goal is to develop an agent that **can continuously operate without human intervention**, autonomously managing its survival, tool durability, and resource efficiency in a dynamic world.

## Meet the Instructor
Plan to meet the instructor by **Week 5** to discuss project progress and refinements.

## AI Tool Usage
We will use the following tools:
- **MineRL**: A reinforcement learning environment for Minecraft.
- **Stable-Baselines3 or RLlib**: For implementing and training the RL agent.
- **Minecraft Modding API**: To integrate crafting and automation mechanics.

All AI tools used will be documented, including their contributions to the project.
