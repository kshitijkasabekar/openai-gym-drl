import gymnasium as gym
from stable_baselines3 import DQN
import time

train_env = gym.make("CartPole-v1")

model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=5e-4,
    buffer_size=100000,
    learning_starts=5000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
)

model.learn(total_timesteps=150000)

model.save("cartpole_sb3")

train_env.close()

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

        time.sleep(0.02)

    print(f"Episode {episode + 1} Reward: {total_reward}")

eval_env.close()