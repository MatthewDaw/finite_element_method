"""Replays memory for DQN."""

import random
from typing import Union

from mesh_generation.mesh_dqn.pydantic_objects import Transition, NonRLTransition


class ReplayMemory:
    """Replays memory for DQN."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def clear(self):
        """Clear memory."""
        self.memory = []
        self.position = 0

    def push(self, transition: Union[Transition, NonRLTransition]):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
