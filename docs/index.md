---
layout: default
title: Home
---

<div class="banner">
  <div class="banner-text">
    <h1>TowerMasters: Reinforcement Learning in Obstacle Tower</h1>
    <p>Welcome to TowerMasters, a CS 175 project where we train a reinforcement learning (RL) agent to navigate the Obstacle Tower environment using Proximal Policy Optimization (PPO). We aim to teach the agent to climb procedurally generated floors by implementing a custom PPO with frame stacking and reward shaping while comparing it to a Stable-Baselines3 baseline.</p>
    <a href="https://github.com/andrewzho/Group-14-oc-rl/tree/main" class="button">Source Code Repository</a>
  </div>
  <div class="banner-image">
    <img src="screenshot2.png" alt="Agent in action">
  </div>
</div>

## Reports
- [Proposal](proposal.md)
- [Status](status.html)
- [Final](final.html)

---

### Source Code Repository
Our project code is hosted on GitHub: [Group-14-oc-rl](https://github.com/andrewzho/Group-14-oc-rl/tree/main).  
- Check out `src/train.py` for our custom PPO  
- `src/train2.py` for the Stable-Baselines3 version

## Project Snapshot
Here's a screenshot of our agent in action during training:

![Agent Training](screenshot.jpg)

## Resources
We've relied on these key resources:
- **Obstacle Tower Environment:** [GitHub](https://github.com/Unity-Technologies/obstacle-tower-env) - The environment we’re tackling.
- **Stable-Baselines3 Documentation:** [Read the Docs](https://stable-baselines3.readthedocs.io/) - For our baseline PPO implementation.
- **PyTorch Documentation:** [Official Docs](https://pytorch.org/docs/stable/index.html) - Core library for our custom PPO.

Stay tuned for our progress report in [status.md](status.md)!

---

Just getting started with Markdown?  
See the [HTML <-> Markdown Quick Reference (Cheat Sheet)][quickref].

[quickref]: https://github.com/mundimark/quickrefs/blob/master/HTML.md
