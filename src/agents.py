import numpy as np


class TabularQLearningAgent:
    """
    Off-policy TD Control: Q-Learning.
    Updates toward the greedy target: max_a Q(s', a).
    """

    def __init__(
        self,
        num_states: int = 96,
        num_actions: int = 3,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.998,
        q_init: float = 10.0,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Optimistic initialization to encourage early exploration
        self.Q = np.full((num_states, num_actions), q_init, dtype=np.float64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        """Off-policy TD update step."""
        best_next_a = np.argmax(self.Q[s_next])
        td_target = r + (0.0 if done else self.gamma * self.Q[s_next, best_next_a])
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.lr * td_error

    def decay_epsilon(self):
        """Anneals exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class TabularSARSAAgent:
    """
    On-policy TD Control: SARSA.
    Updates toward the policy target: Q(s', a') where a' ~ pi(s').
    """

    def __init__(
        self,
        num_states: int = 96,
        num_actions: int = 3,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.998,
        q_init: float = 10.0,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.full((num_states, num_actions), q_init, dtype=np.float64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool):
        """On-policy TD update step."""
        td_target = r + (0.0 if done else self.gamma * self.Q[s_next, a_next])
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.lr * td_error

    def decay_epsilon(self):
        """Anneals exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class ThresholdRuleBaseline:
    """Static enterprise firewall rule heuristic based on telemetry risk score."""

    @staticmethod
    def select_action(state_tuple: tuple[int, int, int, int]) -> int:
        srv, alt, frq, his = state_tuple
        # Risk score calculation
        risk = (alt * 3.0) + (frq * 2.0) + (his * 4.0) + (2.0 if srv in [2, 3] else 0.0)
        if risk >= 9.0:
            return 2  # ISOLATE
        elif risk >= 4.0:
            return 1  # HONEYPOT
        return 0      # PASS


class RandomBaseline:
    """Random agent uniformly sampling discrete actions."""

    @staticmethod
    def select_action() -> int:
        return int(np.random.randint(3))
