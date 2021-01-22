import gym
import time
import torch
import numpy as np
from collections import deque


def train_while_alive(
    agent,
    env,
    n_episodes=20,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.990,
    models_name=("qLocal.pth", "qTarget.pth"),
    info_window=50,
    scale=255,
):

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        dead = False

        while not dead:

            if scale:
                state = np.true_divide(state, scale)

            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, next_state, reward, done)

            lives = info["ale.lives"]
            dead = lives == 0

            if scale:
                next_state = np.true_divide(next_state, scale)

            state = next_state
            score += reward

            if done:
                break

        scores.append(score)
        scores_window.append(score)
        eps = max(eps * eps_decay, eps_end)

        if i_episode % info_window == 0:
            print(
                f"Episode {i_episode}, Average Score {np.mean(scores_window)}, Epsilon: {eps}"
            )

            torch.save(agent.qNetwork_local.state_dict(), models_name[0])
            torch.save(agent.qNetwork_target.state_dict(), models_name[1])
    return scores


def train_by_step(
    agent,
    env,
    n_episodes=2000,
    max_t=2000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.994,
    models_name=("qLocal.pth", "qTarget.pth"),
    info_window=50,
):

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0

        for _ in range(max_t):

            action = agent.act(np.true_divide(state, 255), eps)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, next_state, reward, done)

            state = np.true_divide(next_state, 255)
            score += reward

            if done:
                break

        scores.append(score)
        scores_window.append(score)
        eps = max(eps * eps_decay, eps_end)

        if i_episode % info_window == 0:
            print(
                f"Episode {i_episode}, Average Score {np.mean(scores_window)}, Epsilon: {eps}"
            )

            torch.save(agent.qNetwork_local.state_dict(), models_name[0])
            torch.save(agent.qNetwork_target.state_dict(), models_name[1])
    return scores