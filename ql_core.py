# Imports
import numpy as np
import math as m
from typing import Optional, Union, List, Tuple, Dict, Literal, Any


# Environment Class
class Oriented2DGrid:
    def __init__(
        self,
        grid_size: Tuple[int, int],
        start: Tuple[int, int, float],
        goal: Tuple[int, int, float],
        actions_type: Literal["omni"],
        reward_gains: Dict[str, float] = {
            "goal": 100.0,
            "invalid": 100.0,
            "step": 0.1,
            "turn": 0.1,
            "nearby_obs": 0.1,
        },
        obs_grid: Optional[np.ndarray] = None,
    ) -> None:
        # State space definition
        self._x_size, self._y_size = grid_size
        self._psi_size: int = 8

        # Initial and goal states
        self._start = (start[0], start[1], self._rad2index(start[2]))
        self._goal = (goal[0], goal[1], self._rad2index(goal[2]))
        self._state = self._start

        # Action space definition
        self._angles = np.linspace(0, 2 * m.pi, self._psi_size, endpoint=False)
        self._actions = self._define_actions(actions_type)

        # Reward structure
        self._reward_gains = reward_gains

        # Obstacles grid
        self._obs_grid = obs_grid
        self._nearby_obs_grid = self._precompute_safety_penalty_matrix(min_dist=2.0)

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
            return self._reward_gains["goal"]
        if not self._is_valid_state(current_state, new_state):  # Invalid move
            return -self._reward_gains["invalid"]

        rstep = -self._reward_gains["step"]  # Step penalty

        # Turn Penalty
        turn_angle = abs(action[2])
        rturn = -self._reward_gains["turn"] * turn_angle

        # Obstacle Safety Penalty
        robs = (
            -self._reward_gains["nearby_obs"]
            * self._nearby_obs_grid[new_state[0], new_state[1]]
        )

        # Sum all rewards
        reward = rstep + rturn + robs

        return reward

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

        if not inside_grid:
            return False

        if self._obs_grid is not None:
            # Collision with obstacle
            if self._obs_grid[xf, yf]:
                return False

            # Diagonal corner check
            corner1_is_obstacle = self._obs_grid[x, yf]
            corner2_is_obstacle = self._obs_grid[xf, y]

            if corner1_is_obstacle or corner2_is_obstacle:
                return False

        return True

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
                (0, 0, -1),  # Rotate Counter-Clockwise
                (0, 0, +1),  # Rotate Clockwise
            ]
            return actions
        return []

    def _precompute_safety_penalty_matrix(
        self,
        min_dist: float,
    ) -> np.ndarray:
        if self._obs_grid is None:
            return np.zeros((self._x_size, self._y_size))

        coords = np.where(self._obs_grid == 1)
        obstacle_coords = list(zip(coords[0], coords[1]))

        if not obstacle_coords:
            return np.zeros((self._x_size, self._y_size))

        yy, xx = np.meshgrid(np.arange(self._y_size), np.arange(self._x_size))
        grid_coords = np.stack([xx, yy], axis=-1)

        nearby_obs_matrix = np.zeros((self._x_size, self._y_size), dtype=float)

        for ox, oy in obstacle_coords:
            dist_matrix = np.linalg.norm(grid_coords - (ox, oy), axis=2)
            penalty_for_this_obstacle = min_dist - dist_matrix
            penalty_for_this_obstacle[dist_matrix >= min_dist] = 0.0
            nearby_obs_matrix += penalty_for_this_obstacle

        return nearby_obs_matrix

    def _rad2index(self, angle: float) -> int:
        angle = angle % (2 * m.pi)  # Normalize angle to [0, 2π)
        index = int(round(angle / (2 * m.pi) * self._psi_size)) % self._psi_size
        return index


# Agent Class
class QLAgent:

    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        e_greedy_type: str = "exponential",
        epsilo_start: float = 1.0,
    ) -> None:
        # Hyperparameters for Q-Learning
        self._alpha = learning_rate
        self._gamma = discount_factor
        self._e_greedy_type = e_greedy_type
        self._epsilon_start = epsilo_start
        self.epsilon = self._epsilon_start

        # Q-Table initialization
        self._n_actions = n_actions
        self.q_table = np.zeros(state_shape + (n_actions,))

    def choose_action(self, state: Tuple[int, int, int]) -> int:
        if np.random.random() < self.epsilon:
            # Exploration: select a random action
            return np.random.randint(self._n_actions)
        else:
            # Exploitation: select the best action based on Q-table
            return int(np.argmax(self.q_table[state]))

    def update_q_table(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        new_state: Tuple[int, int, int],
    ) -> None:
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[new_state])

        # Q-Learning formula
        new_value = old_value + self._alpha * (
            reward + self._gamma * next_max - old_value
        )
        self.q_table[state][action] = new_value

    def update_exploration_rate(self, n_episodes: int) -> None:
        # Update epsilon
        min_epsilon = 0.4

        if self.epsilon > min_epsilon:
            if self._e_greedy_type == "linear":
                self.epsilon = self.epsilon - (
                    (self._epsilon_start - min_epsilon) / (n_episodes * 0.8)
                )
            elif self._e_greedy_type == "exponential":
                decay_ratio = (min_epsilon / self._epsilon_start) ** (
                    1 / (n_episodes * 0.8)
                )
                self.epsilon *= decay_ratio
            else:
                raise ValueError("Invalid egreedy type.")

        self.epsilon = max(min_epsilon, self.epsilon)


# Training Function
def train(
    agent: QLAgent,
    environment: Oriented2DGrid,
    n_episodes: int = 20000,
    verbose: bool = False,
    verbose_interval: int = 1000,
) -> Dict[str, List[Any]]:
    data_backup: Dict[str, List[Any]] = {}

    rewards_history: list[float] = []
    epsilon_history: list[float] = []

    max_steps_per_episode = (
        environment.state_shape[0]
        * environment.state_shape[1]
        * environment.state_shape[2]
        * 2
    )

    for episode in range(n_episodes):
        state = environment.reset()
        total_episode_reward = 0.0
        finished = False

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)  # Action selection
            new_state, reward, finished = environment.step(action)  # Environment step
            agent.update_q_table(state, action, reward, new_state)  # Q-Table update

            # State update
            state = new_state
            total_episode_reward += reward

            if finished:  # Episode termination check
                break

        agent.update_exploration_rate(n_episodes)  # Decay exploration rate

        # Individual Data backup update
        rewards_history.append(total_episode_reward)
        epsilon_history.append(agent.epsilon)

        # Console Logging
        if verbose and (episode + 1) % verbose_interval == 0:
            print(f"Episode {episode + 1}/{n_episodes} | Epsilon: {agent.epsilon:.3f}")

    data_backup["rewards_history"] = rewards_history
    data_backup["epsilon_history"] = epsilon_history

    return data_backup
