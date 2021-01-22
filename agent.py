import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random
import numpy as np
import cv2
from utils import ReplayMemory
from models import DQN, DCQN

LR = 5e-4
GAMMA = 0.99
TAU = 1e-3


class Agent:
    """Interacts with and learns from environment"""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        batch_size=64,
        buffer_size=100_000,
        rescale_state=True,
        update_every=20,
        gamma=0.99,
        tau=1e-3,
    ):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.rescale_state = rescale_state
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.rescale = (85, 85)

        if self.rescale_state:
            state_size = self.rescale

        # Q Network (with multi-threading)
        self.qNetwork_local = DCQN(state_size[0], state_size[1], action_size, gray=True)
        self.qNetwork_target = DCQN(
            state_size[0], state_size[1], action_size, gray=True
        )

        # self.qNetwork_local = DQN(state_size, action_size)
        # self.qNetwork_target = DQN(state_size, action_size)

        self.criterion = torch.nn.MSELoss()

        self.optimizer = optim.Adam(self.qNetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayMemory(capacity=self.buffer_size, seed=1)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, next_step, reward, done):
        # Save experience in replay memory
        self.memory.push(state, action, next_step, reward, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if len(self.memory) > self.batch_size and self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            experience = self.memory.sample(self.batch_size)
            self.learn(experience, self.gamma)

    def act(self, state, eps=0):
        """Returns action for given state as per current policy

        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        """
        if self.rescale_state:
            state = cv2.cvtColor(
                cv2.resize(state, self.rescale), cv2.COLOR_BGR2GRAY
            ).astype(np.uint8)
            state = np.expand_dims(state, axis=2)

        state = torch.from_numpy(state).unsqueeze(0).permute(0, 3, 1, 2)

        self.qNetwork_local.eval()

        with torch.no_grad():
            action_values = self.qNetwork_local(state.float())
        self.qNetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, next_states, rewards, dones = experiences

        self.qNetwork_local.train()
        self.qNetwork_target.eval()

        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)
        # Para cada state (frame) do batch, são selecionados somente os Q-values das ações realizadas (gather(1, actions))
        predicted_targets = self.qNetwork_local(states.float()).gather(1, actions)

        # Usando a target Net computamos os Q-values para os states resultantes (next_states) e selecionamos o maior valor para cada state
        with torch.no_grad():
            labels_next = (
                self.qNetwork_target(next_states.float())
                .detach()
                .max(1)[0]
                .unsqueeze(1)
            )

        # Calculamos o "verdadeiro" Q-value
        labels = (rewards + (gamma * labels_next * (1 - dones)))[:, 0].unsqueeze(1)

        # Treinamos a local Net usando back propagation
        loss = self.criterion(predicted_targets, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # self.qNetwork_target.load_state_dict(self.qNetwork_local.state_dict())
        self.soft_update(self.qNetwork_local, self.qNetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1 - tau) * target_param.data
            )


if __name__ == "__main__":
    dqn = DCQN(15, 15, 3)
    randn1 = torch.rand([3, 3, 15, 15])
    randn2 = torch.rand([3, 3, 15, 15])
    print(dqn(randn1))
    # A = Agent((15, 15), 3, 1)

    # for i in range(BATCH_SIZE):
    #    randn1 = torch.rand([1, 3, 15, 15])
    #    randn2 = torch.rand([1, 3, 15, 15])
    #    A.step(randn1, 1, randn2, 0.1, 0)

    Ab = Agent(5, 3, 1)

    # A.learn(A.memory.sample(BATCH_SIZE), GAMMA)
    print(
        Ab.act(
            np.random.rand(
                5,
            )
        )
    )
