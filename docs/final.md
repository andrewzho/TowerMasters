---
layout: default
title: Final Report
---

## Video
<!-- Embed your project video here. The video should be under three minutes, at least 720p, and include a brief problem description, baseline performance, and your best run demonstration. -->
<iframe width="560" height="315" src="https://www.youtube.com/embed/YOUR_VIDEO_ID" frameborder="0" allowfullscreen></iframe>

## Project Summary

The goal of this project is to develop an intelligent agent capable of climbing the Obstacle Tower Challenge—a complex, procedurally generated environment that presents dynamic obstacles, puzzles, and multi-stage challenges. The agent must navigate varying terrains and decision points, making split-second decisions while planning for long-term objectives. This problem is non-trivial as each floor introduces new challenges that require both immediate reaction and strategic foresight.

To tackle this, we leveraged advanced AI/ML techniques tailored for reinforcement learning. We began by establishing a solid baseline using Proximal Policy Optimization (PPO), which provided a robust foundation for the agent’s decision-making process. Recognizing the environment’s inherent complexity, we further enhanced our approach by integrating Long Short-Term Memory (LSTM) networks to capture temporal dependencies, an Intrinsic Curiosity Module (ICM) to encourage effective exploration, and demonstration learning to incorporate expert strategies. These enhancements address the need for both reactive and anticipatory behavior, making the agent more adaptable to the unpredictable nature of the challenge.

By framing the problem within the context of reinforcement learning and advanced neural architectures, our project not only showcases the power of AI in handling complex, dynamic tasks but also highlights the necessity of such methods over traditional rule-based approaches.


## Approaches
### Baseline Methods
- **Proximal Policy Optimization (PPO):** Initially used to establish a reliable baseline. PPO provided stable and promising early results for the agent’s performance.
- **Soft Actor-Critic (SAC):** Also evaluated as a baseline. Although SAC was tested, its performance was ultimately inferior to PPO, which is why the focus shifted towards PPO-based enhancements.

### Enhanced Techniques
- **LSTM Integration:** Incorporated Long Short-Term Memory networks to capture temporal dependencies, enabling the agent to better utilize historical data for decision-making.
- **ICM Curriculum:** Applied an Intrinsic Curiosity Module to encourage enhanced exploration through intrinsic rewards, helping the agent navigate complex parts of the tower.
- **Demonstration Learning:** Leveraged expert demonstrations to guide the learning process, thus providing a structured way to learn effective policies.

These enhancements built upon the PPO baseline led to significant improvements in the agent’s ability to successfully climb the Obstacle Tower Challenge.

## Evaluation
Evaluation was conducted using both quantitative and qualitative methods:
- **Quantitative Metrics:** Success rates, time to completion, and cumulative rewards were recorded to compare the performance of the PPO and SAC baselines alongside the enhanced PPO model.
- **Qualitative Analysis:** Video captures and behavioral observations were used to assess improvements in navigation and strategy, particularly after integrating LSTM, ICM, and demonstration learning.

These evaluations confirmed that the enhanced PPO approach significantly outperformed the other methods, validating the choices made in the project.

## References
- Schulman, J., et al. "Proximal Policy Optimization Algorithms." (2017)
- Haarnoja, T., et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." (2018)
- Pathak, D., et al. "Curiosity-driven Exploration by Self-supervised Prediction." (2017)
