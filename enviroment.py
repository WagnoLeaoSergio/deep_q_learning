import gym
import time
import torch
import numpy as np
from agent import Agent
from collections import deque
import matplotlib.pyplot as plt

from training import train_while_alive, train_by_step

ENV_NAME = "Pong-v0"
AGENT_ACTIONS = 6


def plot_score(scores, img_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.savefig(f"{img_name}.png")


def show(agent, num_eps=1000, time_stamp=0, scale=0):
    state = env.reset()
    # done = False

    local_checkpoint = torch.load("qLocal.pth", map_location=torch.device("cpu"))
    target_checkpoint = torch.load("qTarget.pth", map_location=torch.device("cpu"))

    agent.qNetwork_local.load_state_dict(local_checkpoint)
    agent.qNetwork_target.load_state_dict(target_checkpoint)

    actions_count = 0

    for _ in range(num_eps):

        env.render()

        # action = env.action_space.sample()
        # next_state, reward, done, info = env.step(action)
        if scale:
            state = np.true_divide(state, scale)

        if not actions_count:
            action = agent.act(state, 0)

        actions_count = (actions_count + 1) % 3

        next_state, reward, done, info = env.step(action)

        if scale:
            next_state = np.true_divide(state, scale)

        state = next_state

        if time_stamp:
            time.sleep(time_stamp)

        if done:
            state = env.reset()
            break

    env.close()


if __name__ == "__main__":
    env = gym.make("MsPacman-v0")
    agentDCQN = Agent(state_size=(210, 160), action_size=9, seed=1)

    # scores = train_by_step(
    #     agentDCQN, env, n_episodes=10, max_t=1000, eps_decay=0.98, info_window=1
    # )

    # plot_score(scores, "pong_scores")
    show(agentDCQN, num_eps=6000, time_stamp=0.09)
