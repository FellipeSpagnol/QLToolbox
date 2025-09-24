# Imports
import numpy as np
import math as m
from typing import Optional, Union, List, Tuple, Dict, Literal


# Class Definition
class Environment:
    def __init__(
        self,
        grid_size: Tuple[int, int],
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        actions_type: Literal["omni"],
    ) -> None:
        # State space definition
        self._x_size, self._y_size = grid_size
        self._psi_size: int = 8

        # Initial and goal states
        self._start = start
        self._goal = goal
        self._state = self._start

        # Action space definition
        self._angles = np.linspace(0, 2 * m.pi, self._psi_size, endpoint=False)
        self._actions = self._define_actions(actions_type)

    @property
    def state_shape(self) -> Tuple[int, int, int]:
        return (self._x_size, self._y_size, self._psi_size)

    @property
    def n_actions(self) -> int:
        return len(self._actions)

    def step(self, action_index: int) -> Tuple[Tuple[int, int, int], float, bool]:
        current_state = self._state
        action = self._actions[action_index]

        new_state = self._calculate_new_state(
            current_state, action
        )  # Calculate potential new state

        reward = self._calculate_reward(
            current_state, action, new_state
        )  # Calculate reward considering potential new state

        self._state = (
            new_state
            if self._is_valid_state(current_state, new_state)
            else current_state
        )  # Update state if the potential new state is valid

        finished = self._state == self._goal  # Check if the episode is task is finished

        return self._state, reward, finished

    def reset(self) -> Tuple[int, int, int]:
        self._state = self._start
        return self._state

    def _calculate_reward(
        self,
        current_state: Tuple[int, int, int],
        action: Tuple[int, int, int],
        new_state: Tuple[int, int, int],
    ) -> float:

        if new_state == self._goal:  # Goal reached
            return 1.0
        if not self._is_valid_state(current_state, new_state):  # Invalid move
            return -1.0
        else:  # Valid move
            return -0.01

    def _is_valid_state(
        self,
        current_state: Tuple[int, int, int],
        new_state: Tuple[int, int, int],
    ) -> bool:
        x, y, psi = current_state
        xf, yf, psif = new_state

        inside_grid = (0 <= xf < self._x_size) and (
            0 <= yf < self._y_size
        )  # Check grid boundaries

        return inside_grid

    def _calculate_new_state(
        self,
        current_state: Tuple[int, int, int],
        action: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:

        x, y, psi = current_state
        dx, dy, dpsi = action

        new_x = x + dx
        new_y = y + dy
        new_psi = (psi + dpsi) % self._psi_size

        return (new_x, new_y, new_psi)

    def _define_actions(self, actions_type: str) -> List[Tuple[int, int, int]]:
        if actions_type == "omni":  # Actions for omnidirectional movement
            actions = [
                (1, 0, 0),  # Right
                (-1, 0, 0),  # Left
                (0, 1, 0),  # Up
                (0, -1, 0),  # Down
                (1, 1, 0),  # Right Up
                (-1, -1, 0),  # Left Down
                (1, -1, 0),  # Right Down
                (-1, 1, 0),  # Left Up
                (0, 0, +1),  # Rotate Clockwise
                (0, 0, -1),  # Rotate Counter-Clockwise
            ]
            return actions
        return []


class Agent:
    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay_rate: float = 0.995,
        min_epsilon: float = 0.01,
    ) -> None:
        # Hyperparameters for Q-Learning
        self._alpha = learning_rate
        self._gamma = discount_factor
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay_rate
        self._epsilon_min = min_epsilon

        # Q-Table initialization
        self._n_actions = n_actions
        self._q_table = np.zeros(state_shape + (n_actions,))

    def choose_action(self, state: Tuple[int, int, int]) -> int:
        if np.random.random() < self._epsilon:
            # Exploration: select a random action
            return np.random.randint(self._n_actions)
        else:
            # Exploitation: select the best action based on Q-table
            return int(np.argmax(self._q_table[state]))

    def update_q_table(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        new_state: Tuple[int, int, int],
    ) -> None:
        """
        Updates the Q-value for a given state-action pair using the Bellman equation.
        """
        old_value = self._q_table[state][action]
        next_max = np.max(self._q_table[new_state])

        # Q-Learning formula
        new_value = old_value + self._alpha * (
            reward + self._gamma * next_max - old_value
        )
        self._q_table[state][action] = new_value

    def update_exploration_rate(self) -> None:
        """
        Decays the exploration rate (epsilon) to reduce exploration over time.
        The rate does not fall below a minimum value.
        """
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)


# Training Function
def train(agent: Agent, environment: Environment) -> None:
    pass
