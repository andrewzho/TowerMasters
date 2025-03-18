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

Ran tool

Ran tool

Search files...

Search files...

Search files...

Read file: README.md
## Approaches

### Baseline Approaches

#### Vanilla PPO (Proximal Policy Optimization)
Our baseline approach implements the Proximal Policy Optimization algorithm as introduced by Schulman et al. (2017). PPO is a policy gradient method that optimizes a clipped surrogate objective function:

```
L^CLIP(θ) = Êt[min(rt(θ)Ât, clip(rt(θ), 1-ε, 1+ε)Ât)]
```

**Advantages:**
- Stable policy updates through clipping mechanism
- Good sample efficiency compared to other policy gradient methods
- Straightforward implementation with fewer hyperparameters than some alternatives

**Disadvantages:**
- Limited ability to handle environments requiring temporal reasoning
- Struggles with sparse rewards in complex 3D environments
- Relies on random exploration without intrinsic motivation

#### Stable Baselines3 PPO
We also implemented a baseline using the Stable Baselines3 library's PPO implementation (`train_sb.py`).

**Advantages:**
- Production-ready implementation with optimized performance
- Built-in monitoring tools and callbacks
- Simple training pipeline requiring minimal code

**Disadvantages:**
- Less flexibility for customization
- Limited support for recurrent policies and demonstration learning
- Harder to integrate with complex exploration strategies

### Proposed Approaches

#### LSTM-Enhanced PPO
To handle the temporal dependencies in Obstacle Tower (remembering key locations, puzzle solutions, etc.), we extended PPO with LSTM layers:

```
1. Process observations through CNN layers to extract visual features
2. Feed visual features into LSTM layers to maintain temporal context
3. Use final hidden state to output policy and value
4. For training:
   a. Process trajectories as sequences of length L
   b. Reset LSTM states at episode boundaries
   c. Maintain LSTM states between timesteps during rollouts
```

The model architecture (`RecurrentPPONetwork` in `model.py`) includes:
- CNN layers with residual connections and attention mechanisms
- Two-layer LSTM network with 256 hidden units
- Dual-stream output for policy and value functions

**Advantages:**
- Better memory for tracking previously visited areas and puzzle elements
- Improved navigation across floors by remembering layouts
- Can learn complex temporal dependencies in the environment

**Disadvantages:**
- More computationally expensive than non-recurrent approaches
- Requires careful sequence handling during training
- More challenging to stabilize during optimization

#### Demonstration-Guided PPO (DemoPPO)
To overcome exploration challenges, we implemented a demonstration-enhanced PPO (`DemoPPO` class in `ppo.py`) that incorporates behavioral cloning from expert demonstrations:

```
L_total = L_PPO + λ_BC * L_BC

where L_BC = -log(π(a_demo|s_demo))
```

**Implementation:**
```
1. Load expert demonstrations (keyframe sequences showing successful gameplay)
2. During PPO updates:
   a. Sample demonstration batch B_demo from demonstration buffer
   b. Compute standard PPO loss on online experience
   c. Compute behavior cloning loss on demonstrations
   d. Apply both gradients with appropriate weighting
```

**Advantages:**
- Dramatically accelerates learning by leveraging expert knowledge
- Helps agent discover sparse rewards that are difficult to find randomly
- Provides a behavioral prior that guides policy development

**Disadvantages:**
- Can lead to mimicry instead of understanding
- Performance limited by demonstration quality
- May limit exploration beyond demonstrated behaviors

#### Intrinsic Curiosity Module (ICM)
To further improve exploration, we implemented an Intrinsic Curiosity Module (`ICM` class in `icm.py`) that generates curiosity-driven intrinsic rewards:

```
1. Train a forward dynamics model f that predicts next state features from current state features and actions
2. Train an inverse dynamics model g that predicts actions from current and next state features
3. Generate intrinsic reward proportional to forward model prediction error:
   r_intrinsic = η/2 * ||f(st, at) - φ(st+1)||²
```

**Advantages:**
- Drives exploration toward novel states even with sparse external rewards
- Particularly effective in visually diverse environments
- Complements demonstration learning by encouraging exploration beyond demonstrations

**Disadvantages:**
- Can be distracted by stochastic elements in the environment
- Adds computational overhead
- Requires careful tuning of intrinsic reward scale

#### Full Approach: Recurrent-DemoPPO with ICM
Our complete approach combines all three enhancements:

```
1. Process observations through RecurrentPPONetwork with LSTM layers
2. Update policy using demonstration data with behavior cloning loss
3. Generate intrinsic rewards using ICM for enhanced exploration
4. Total loss function:
   L_total = L_PPO + λ_BC * L_BC + (L_forward + L_inverse)_ICM
```

Key hyperparameters:
- LSTM hidden size: 256
- Sequence length for recurrent training: 16
- Demonstration batch size: 64
- Behavior cloning weight: 0.1 (annealed over time)
- ICM reward scale: 0.01

**Advantages:**
- Comprehensive solution addressing all major challenges:
  - Memory (LSTM)
  - Exploration (ICM)
  - Sample efficiency (demonstrations)
- Strongest performance on higher floors with complex puzzles
- Better generalization to unseen procedural layouts

**Disadvantages:**
- Most complex implementation requiring careful integration of components
- Highest computational requirements
- Multiple interacting hyperparameters requiring careful tuning

Our experiments showed that this combined approach significantly outperformed the baseline methods, particularly in advancing to higher floors and handling complex puzzle elements in the Obstacle Tower environment.


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
