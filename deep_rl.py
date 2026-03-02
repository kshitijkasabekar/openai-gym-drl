import gymnasium as gym
from stable_baselines3 import DQN
import time

# -------------------------
# 1️⃣ TRAINING (No Rendering)
# -------------------------
train_env = gym.make("CartPole-v1")

model = DQN(
    "MlpPolicy",
    train_env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=250
)

model.learn(total_timesteps=50000)

model.save("cartpole_sb3")

train_env.close()


# -------------------------
# 2️⃣ VISUALIZATION (With Graphics)
# -------------------------
eval_env = gym.make("CartPole-v1", render_mode="human")

model = DQN.load("cartpole_sb3")

episodes = 3

for episode in range(episodes):
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Slow down so you can see it clearly
        time.sleep(0.02)

    print(f"Episode {episode + 1} Reward: {total_reward}")

eval_env.close()