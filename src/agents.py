import random
import numpy as np


class BaseTabularAgent:
    """Shared functionality for tabular TD control agents."""

    def __init__(
        self,
        num_states: int = 96,
        num_actions: int = 3,
        lr: float = 0.1,
        lr_min: float = 0.005,
        lr_decay: float = 0.998,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9965,
        q_init: float = 10.0,
        seed: int = None,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)

        # Optimistic initialization to encourage early exploration
        self.Q = np.full((num_states, num_actions), q_init, dtype=np.float64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.num_actions - 1)
        return int(np.argmax(self.Q[state]))

    def decay_parameters(self):
        """Anneals both exploration rate (epsilon) and learning rate (alpha)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.lr = max(self.lr_min, self.lr * self.lr_decay)

    def decay_epsilon(self):
        """Alias for backward compatibility with Milestone 3 scripts."""
        self.decay_parameters()

    def save(self, filepath: str):
        """Persists the learned Q-table to disk."""
        np.save(filepath, self.Q)

    def load(self, filepath: str):
        """Loads a saved Q-table from disk."""
        self.Q = np.load(filepath)


class TabularQLearningAgent(BaseTabularAgent):
    """
    Off-policy TD Control: Q-Learning.
    Updates toward the greedy target: max_a Q(s', a).
    """

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        best_next_a = int(np.argmax(self.Q[s_next]))
        td_target = r + (0.0 if done else self.gamma * self.Q[s_next, best_next_a])
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.lr * td_error


class TabularSARSAAgent(BaseTabularAgent):
    """
    On-policy TD Control: SARSA.
    Updates toward the policy target: Q(s', a') where a' ~ pi(s').
    """

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool):
        td_target = r + (0.0 if done else self.gamma * self.Q[s_next, a_next])
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.lr * td_error


class ThresholdRuleBaseline:
    """Static enterprise firewall rule heuristic based on telemetry risk score."""

    @staticmethod
    def select_action(state_tuple: tuple[int, int, int, int]) -> int:
        srv, alt, frq, his = state_tuple
        risk = (alt * 3.0) + (frq * 2.0) + (his * 4.0) + (2.0 if srv in (2, 3) else 0.0)
        if risk >= 9.0:
            return 2  # ISOLATE
        elif risk >= 4.0:
            return 1  # HONEYPOT
        return 0      # PASS


class RandomBaseline:
    """Random agent uniformly sampling discrete actions."""

    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def select_action(self) -> int:
        return self.rng.randint(0, 2)