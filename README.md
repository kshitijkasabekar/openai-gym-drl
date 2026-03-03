# openai-gym-drl
# 🎮 Deep Reinforcement Learning – CartPole with DQN

This project implements a Deep Q-Network (DQN) agent to solve the classic **CartPole-v1** environment using Stable-Baselines3.

The agent learns to balance a pole on a moving cart by interacting with the environment and optimizing long-term reward.

---

## 📌 Environment

- Environment: CartPole-v1
- Framework: Gymnasium
- Algorithm: Deep Q-Network (DQN)
- RL Library: Stable-Baselines3
- Backend: PyTorch

---

## 🧠 Problem Description

CartPole is a classic control problem where:

- A pole is attached to a cart by an unactuated joint.
- The cart moves along a frictionless track.
- The goal is to prevent the pole from falling.

### State Space (4-dimensional):
1. Cart position
2. Cart velocity
3. Pole angle
4. Pole angular velocity

### Action Space (Discrete):
- `0` → Push cart left
- `1` → Push cart right

### Reward:
+1 for every timestep the pole remains balanced.

The episode terminates if:
- Pole angle exceeds threshold
- Cart position exceeds boundary
- 500 steps are reached

---
