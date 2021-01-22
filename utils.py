import random
import numpy as np
import torch
from collections import namedtuple, deque
import cv2


class ReplayMemory(object):
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.seed = random.seed(seed)
        self.transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "done")
        )

    def push(self, state, action, next_state, reward, done):
        """Saves a transition."""
        if self.rescale:
            state = cv2.cvtColor(
                cv2.resize(state, (85, 85)), cv2.COLOR_BGR2GRAY
            ).astype(np.uint8)

            next_state = cv2.cvtColor(
                cv2.resize(next_state, (85, 85)), cv2.COLOR_BGR2GRAY
            ).astype(np.uint8)

        if len(self.memory) < self.capacity:
            self.memory.append(self.transition(state, action, next_state, reward, done))
        self.memory[self.position] = self.transition(
            state, action, next_state, reward, done
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        experiences = random.sample(self.memory, k=batch_size)
        states, actions, next_states, rewards, dones = zip(*experiences)

        states = torch.from_numpy(np.expand_dims(np.array(list(states)), axis=3))
        actions = torch.from_numpy(np.array(list(actions))).unsqueeze(1).long()
        next_states = torch.from_numpy(
            np.expand_dims(np.array(list(next_states)), axis=3)
        )
        rewards = torch.from_numpy(np.array(list(rewards))).float()
        dones = torch.from_numpy(np.array(list(dones))).int()

        return (states, actions, next_states, rewards, dones)

    def __len__(self):
        return len(self.memory)