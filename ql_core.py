# Imports
import numpy as np
import math as m
from typing import Optional, Union, List, Tuple, Dict, Literal, Any, TypedDict


class StateDict(TypedDict):
    navigation: Tuple[int, int, int]
    safety: Tuple[int, ...]


# Environment Class
class Oriented2DGrid:
    def __init__(
        self,
        grid_size: Tuple[int, int],
        start: Tuple[int, int, float],
        goal: Tuple[int, int, float],
        actions_type: Literal["omni", "diff"],
        reward_gains: Dict[str, float] = {
            "goal": 100.0,
            "invalid": 100.0,
            "move": 0.1,
            "turn": 0.1,
            "nearby_obs": 0.1,
        },
        obs_grid: Optional[np.ndarray] = None,
        random_start_percentage: float = 0.6,
    ) -> None:
        # Navigation State space definition
        self._x_size, self._y_size = grid_size
        self._psi_size: int = 8

        # Safety State space definition
        self._sensor_lines: int = 8
        self._sensor_layers: int = 3

        # Initial and goal states
        self._start = (start[0], start[1], self._rad2index(start[2]))
        self._goal = (goal[0], goal[1], self._rad2index(goal[2]))

        self._state: StateDict = {
            "navigation": self._start,
            "safety": tuple(0 for _ in range(self._sensor_lines)),
        }

        # Action space definition
        self._angles = np.linspace(0, 2 * m.pi, self._psi_size, endpoint=False)
        self._actions_type = actions_type

        # Reward structure
        self._reward_gains = reward_gains

        # Obstacles grid
        self._obs_grid = obs_grid

        # Precompute auxiliary data
        self._nearby_obs_grid = self._precompute_safety_penalty_matrix(min_dist=2.0)
        self._valid_start_positions = (
            np.argwhere(self._obs_grid == 0)
            if self._obs_grid is not None
            else [(x, y) for x in range(self._x_size) for y in range(self._y_size)]
        )

        # Other parameters
        self._random_start_percentage = random_start_percentage

    @property
    def state_shape(self) -> Dict[str, Tuple[int, ...]]:
        navigation_shape = (self._x_size, self._y_size, self._psi_size)
        safety_shape = tuple([self._sensor_layers] * self._sensor_lines)

        return {
            "navigation": navigation_shape,
            "safety": safety_shape,
        }

    @property
    def n_actions(self) -> int:
        return len(self._actions)

    @property
    def _actions(self) -> List[Tuple[int, int, int]]:
        def rint(x):
            return int(np.round(x))

        angle = self._index2rad(self._state["navigation"][2])

        if self._actions_type == "omni":  # Actions for omnidirectional movement
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
        elif self._actions_type == "diff":  # Actions for differential drive movement
            actions = [
                (rint(m.cos(angle)), rint(m.sin(angle)), 0),
                (-rint(m.cos(angle)), -rint(m.sin(angle)), 0),
                (0, 0, +1),
                (0, 0, -1),
            ]
        return actions

    def step(self, action_index: int) -> Tuple[StateDict, Dict[str, float], bool]:
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

    def reset(self) -> StateDict:
        if np.random.random() < self._random_start_percentage:
            random_index = np.random.randint(len(self._valid_start_positions))
            xr, yr = self._valid_start_positions[random_index]
            psir = np.random.randint(self._psi_size)

            random_position = (xr, yr, psir)
            equivalent_sensor_values = self._get_sensor_values(random_position)

            self._state = {
                "navigation": random_position,
                "safety": equivalent_sensor_values,
            }
            return self._state
        else:
            equivalent_sensor_values = self._get_sensor_values(self._start)
            self._state = {
                "navigation": self._start,
                "safety": equivalent_sensor_values,
            }
            return self._state

    def _calculate_reward(
        self,
        current_state: StateDict,
        action: Tuple[int, int, int],
        new_state: StateDict,
    ) -> Dict[str, float]:
        # Navigation Reward
        if new_state["navigation"] == self._goal:  # Goal reached
            navigation_reward = self._reward_gains["goal"]
        elif not self._is_valid_state(current_state, new_state)[
            "navigation"
        ]:  # Invalid move (out of bounds)
            navigation_reward = self._reward_gains["invalid"]
        else:
            # Move Penalty
            rmove = -self._reward_gains["move"] * float(
                np.linalg.norm([action[0], action[1]])
            )

            # Turn Penalty
            turn_angle = abs(action[2])
            rturn = -self._reward_gains["turn"] * turn_angle

            # Full Reward
            navigation_reward = rmove + rturn

        # Safety Reward
        if not self._is_valid_state(current_state, new_state)[
            "safety"
        ]:  # Invalid move (collision)
            safety_reward = -self._reward_gains["invalid"]
        else:
            # Obstacle Safety Penalty
            robs = (
                -self._reward_gains["nearby_obs"]
                * self._nearby_obs_grid[
                    new_state["navigation"][0], new_state["navigation"][1]
                ]
            )

            # Full Reward
            safety_reward = robs

        # Full reward dict
        reward_dict = {
            "navigation": navigation_reward,
            "safety": safety_reward,
        }

        return reward_dict

    def _is_valid_state(
        self,
        current_state: StateDict,
        new_state: StateDict,
    ) -> Dict[str, bool]:

        # Navigation validity check
        x, y, psi = current_state["navigation"]
        xf, yf, psif = new_state["navigation"]

        inside_grid = (0 <= xf < self._x_size) and (
            0 <= yf < self._y_size
        )  # Check grid boundaries

        # Safety
        if self._obs_grid is not None:
            # # Collision with obstacle
            # if self._obs_grid[xf, yf]:
            #     return False

            # # Diagonal corner check
            # corner1_is_obstacle = self._obs_grid[x, yf]
            # corner2_is_obstacle = self._obs_grid[xf, y]

            # if corner1_is_obstacle or corner2_is_obstacle:
            #     return False
            no_collision = True

        return {"navigation": inside_grid, "safety": no_collision}

    def _calculate_new_state(
        self,
        current_state: StateDict,
        action: Tuple[int, int, int],
    ) -> StateDict:

        x, y, psi = current_state["navigation"]
        dx, dy, dpsi = action

        new_x = x + dx
        new_y = y + dy
        new_psi = (psi + dpsi) % self._psi_size
        new_position = (new_x, new_y, new_psi)

        new_sensor_values = self._get_sensor_values(new_position)

        return {"navigation": new_position, "safety": new_sensor_values}

    def _get_sensor_values(self, position: Tuple[int, int, int]) -> Tuple[int, ...]:
        # Note: Test version (made by gemini)
        current_x, current_y, current_theta_idx = position
        readings = []

        # Sensor anlges
        sensor_angles = [0, 1, 2, 3, 4, 5, 6, 7]
        max_range = 5

        for offset in sensor_angles:
            ray_dir_idx = (current_theta_idx + offset) % 8
            angle_rad = self._index2rad(ray_dir_idx)

            dx = int(np.round(np.cos(angle_rad)))
            dy = int(np.round(np.sin(angle_rad)))

            dist = max_range

            for r in range(1, max_range + 1):
                check_x = current_x + (dx * r)
                check_y = current_y + (dy * r)

                if (
                    check_x < 0
                    or check_x >= self._x_size
                    or check_y < 0
                    or check_y >= self._y_size
                ):
                    dist = r
                    break
                if self._obs_grid is not None and self._obs_grid[check_x, check_y] == 1:
                    dist = r
                    break

            if dist <= 1:
                val = 2
            elif dist <= 3:
                val = 1
            else:
                val = 0

            readings.append(val)

        return tuple(readings)

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

    def _index2rad(self, index: int) -> float:
        index = index % self._psi_size
        angle = (index / self._psi_size) * (2 * m.pi)
        return angle


# Agent Class
class QLAgent:

    def __init__(
        self,
        state_shape: Dict[str, Tuple[int, ...]],
        n_actions: int,
        learning_rate: float = 0.2,
        discount_factor: float = 0.9,
        e_greedy_type: str = "exponential",
        epsilon_start: float = 1.0,
    ) -> None:
        # Hyperparameters for Q-Learning
        self._alpha = learning_rate
        self._gamma = discount_factor
        self._e_greedy_type = e_greedy_type
        self._epsilon_start = epsilon_start
        self.epsilon = self._epsilon_start

        # Q-Table initialization
        self._n_actions = n_actions
        self._q_table = {
            "navigation": np.zeros(state_shape["navigation"] + (n_actions,)),
            "safety": np.zeros(state_shape["safety"] + (n_actions,)),
        }

    def choose_action(self, state: StateDict) -> int:
        if np.random.random() < self.epsilon:
            # Exploration: select a random action
            return np.random.randint(self._n_actions)
        else:
            # Exploitation: select the best action based on Q-table
            total_q_values = np.zeros(self._n_actions)

            for key in self._q_table.keys():
                sub_state = state[key]  # type: ignore
                q_values = self._q_table[key][sub_state]
                total_q_values += q_values

            return int(np.argmax(total_q_values))

    def update_q_table(
        self,
        state: StateDict,
        action: int,
        reward: Dict[str, float],
        new_state: StateDict,
    ) -> None:
        for key in self._q_table.keys():
            agent_q_table = self._q_table[key]
            agent_reward = reward[key]
            agent_state = state[key]  # type: ignore
            agent_new_state = new_state[key]  # type: ignore
            old_value = agent_q_table[agent_state][action]
            next_max_value = np.max(agent_q_table[agent_new_state])

            # Q-Learning formula for updating individual Q-value
            new_value = old_value + self._alpha * (
                agent_reward + self._gamma * next_max_value - old_value
            )
            agent_q_table[agent_state][action] = new_value

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
    verbose: bool = True,
    verbose_interval: int = 1000,
    enable_early_stopping: bool = True,
    early_stopping_criterion: str = "reward_plateau",
    early_stop_window_size: int = 500,
    early_stop_min_improvement: float = 0.1,
    early_stop_patience: int = 3,
    q_delta_threshold: float = 1e-4,
    q_delta_check_interval: int = 100,
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

    patience_counter = 0
    q_table_old = (
        agent._q_table.copy()
        if enable_early_stopping and early_stopping_criterion == "q_table_stability"
        else None
    )

    for episode in range(n_episodes):
        state = environment.reset()
        total_episode_reward = 0.0
        finished = False

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            new_state, reward, finished = environment.step(action)
            agent.update_q_table(state, action, reward, new_state)
            state = new_state
            total_episode_reward += sum(reward.values())
            if finished:
                break

        agent.update_exploration_rate(n_episodes)
        rewards_history.append(total_episode_reward)
        epsilon_history.append(agent.epsilon)

        if verbose and (episode + 1) % verbose_interval == 0:
            print(f"Episode {episode + 1}/{n_episodes} | Epsilon: {agent.epsilon:.3f}")

        if enable_early_stopping:
            if (
                early_stopping_criterion == "reward_plateau"
                and episode >= 2 * early_stop_window_size
            ):
                recent_avg_reward = np.mean(rewards_history[-early_stop_window_size:])
                previous_avg_reward = np.mean(
                    rewards_history[
                        -2 * early_stop_window_size : -early_stop_window_size
                    ]
                )
                improvement = recent_avg_reward - previous_avg_reward
                if improvement < early_stop_min_improvement:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= early_stop_patience:
                    print(
                        f"\n--- Early Stopping (Reward Plateau) at Episode {episode + 1} ---"
                    )
                    print(f"Average reward has not improved sufficiently.")
                    break

            elif (
                early_stopping_criterion == "q_table_stability"
                and (episode + 1) % q_delta_check_interval == 0
            ):
                if q_table_old is not None:
                    delta = np.sum(np.abs(agent._q_table - q_table_old))
                else:
                    delta = float("inf")
                q_table_old = agent._q_table.copy()

                if delta < q_delta_threshold:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= early_stop_patience:
                    print(
                        f"\n--- Early Stopping (Q-Table Stability) at Episode {episode + 1} ---"
                    )
                    print(f"Q-Table has converged. Total delta: {delta:.6f}")
                    break

    data_backup["rewards_history"] = rewards_history
    data_backup["epsilon_history"] = epsilon_history

    return data_backup
