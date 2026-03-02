import random
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')

episodes = 20

for episode in range(1, episodes + 1):
    state, _ = env.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 1])
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward

    print(f"Episode: {episode}, Score: {score}")

env.close()