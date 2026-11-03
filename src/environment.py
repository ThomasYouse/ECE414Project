import numpy as np


class NetDefendEnv:
    """
    Adaptive Intrusion Detection and Dynamic Honeypot Routing Simulator.
    Models an enterprise entrance gateway protecting 4 production services
    (Web, Auth, Database, Bastion) backed by a pooled honeypot buffer.
    """

    def __init__(self, honeypot_capacity: int = 5, max_steps: int = 200, benign_ratio: float = 0.70, seed: int = None):
        self.honeypot_capacity = honeypot_capacity
        self.max_steps = max_steps
        self.benign_ratio = benign_ratio

        # State dimensions: 4 services x 4 alert levels x 3 request frequencies x 2 IP histories = 96 states
        self.num_services = 4
        self.num_alerts = 4
        self.num_frequencies = 3
        self.num_histories = 2
        self.num_states = 96
        self.num_actions = 3  # 0: PASS, 1: HONEYPOT, 2: ISOLATE

        # Attacker hidden progression: Recon (0) -> Lateral (1) -> Exfil (2)
        self.p_att = np.array([
            [0.60, 0.35, 0.05],
            [0.10, 0.65, 0.25],
            [0.00, 0.20, 0.80]
        ], dtype=np.float64)

        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    @staticmethod
    def encode_state(service: int, alert: int, freq: int, history: int) -> int:
        """Encodes state 4-tuple into a discrete integer index in [0, 95]."""
        return int(24 * service + 6 * alert + 2 * freq + history)

    @staticmethod
    def decode_state(idx: int) -> tuple[int, int, int, int]:
        """Decodes discrete integer index in [0, 95] back to state 4-tuple."""
        history = idx % 2
        idx //= 2
        freq = idx % 3
        idx //= 3
        alert = idx % 4
        service = idx // 4
        return (service, alert, freq, history)

    def reset(self, seed: int = None) -> tuple[int, dict]:
        """Resets the environment to initial conditions."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.active_honeypots = []  # Tracks remaining step durations for allocated decoys
        self.attacker_stage = 0     # 0: Recon, 1: Lateral Movement, 2: Exfiltration
        self.current_flow_is_malicious = False

        self.current_state_tuple = self._generate_inbound_flow()
        self.current_state_idx = self.encode_state(*self.current_state_tuple)

        info = {
            "step": self.current_step,
            "honeypot_load": len(self.active_honeypots),
            "attacker_stage": self.attacker_stage,
            "is_malicious": self.current_flow_is_malicious,
            "state_tuple": self.current_state_tuple
        }
        return self.current_state_idx, info

    def _generate_inbound_flow(self) -> tuple[int, int, int, int]:
        """Emits observable flow metadata conditioned on underlying traffic type and attacker stage."""
        self.current_flow_is_malicious = bool(self.rng.random() > self.benign_ratio)

        if not self.current_flow_is_malicious:
            # Benign distribution: heavily Web/Auth, low alerts, clean IP history
            service = int(self.rng.choice([0, 1, 2, 3], p=[0.55, 0.30, 0.10, 0.05]))
            alert = int(self.rng.choice([0, 1, 2, 3], p=[0.75, 0.18, 0.06, 0.01]))
            freq = int(self.rng.choice([0, 1, 2], p=[0.80, 0.17, 0.03]))
            history = int(self.rng.choice([0, 1], p=[0.96, 0.04]))
        else:
            # Progress attacker's hidden Markov chain
            self.attacker_stage = int(self.rng.choice([0, 1, 2], p=self.p_att[self.attacker_stage]))

            if self.attacker_stage == 0:  # Reconnaissance
                service = int(self.rng.choice([0, 1, 2, 3], p=[0.40, 0.30, 0.10, 0.20]))
                alert = int(self.rng.choice([0, 1, 2, 3], p=[0.20, 0.50, 0.25, 0.05]))
                freq = int(self.rng.choice([0, 1, 2], p=[0.30, 0.50, 0.20]))
                history = int(self.rng.choice([0, 1], p=[0.70, 0.30]))
            elif self.attacker_stage == 1:  # Lateral Movement
                service = int(self.rng.choice([0, 1, 2, 3], p=[0.10, 0.35, 0.35, 0.20]))
                alert = int(self.rng.choice([0, 1, 2, 3], p=[0.05, 0.25, 0.50, 0.20]))
                freq = int(self.rng.choice([0, 1, 2], p=[0.20, 0.60, 0.20]))
                history = int(self.rng.choice([0, 1], p=[0.40, 0.60]))
            else:  # Exfiltration
                service = int(self.rng.choice([0, 1, 2, 3], p=[0.05, 0.05, 0.75, 0.15]))
                alert = int(self.rng.choice([0, 1, 2, 3], p=[0.02, 0.08, 0.35, 0.55]))
                freq = int(self.rng.choice([0, 1, 2], p=[0.10, 0.30, 0.60]))
                history = int(self.rng.choice([0, 1], p=[0.15, 0.85]))

        return (service, alert, freq, history)

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """Applies routing action, decays honeypot sessions, and computes reward."""
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action {action}. Action must be in {{0: PASS, 1: HONEYPOT, 2: ISOLATE}}")

        self.current_step += 1

        # 1. Age active honeypots and release expired sessions
        self.active_honeypots = [dur - 1 for dur in self.active_honeypots if dur - 1 > 0]
        honeypot_saturated = len(self.active_honeypots) >= self.honeypot_capacity

        reward = 0.0
        overflow = False
        is_mal = self.current_flow_is_malicious

        # 2. Evaluate reward matrix and state updates
        if action == 0:  # PASS
            reward = -100.0 if is_mal else +5.0

        elif action == 1:  # HONEYPOT
            if honeypot_saturated:
                overflow = True
                # Buffer exhausted: connection leaks into production / drops
                reward = -120.0 if is_mal else -10.0
            else:
                # Stochastic holding duration: Geometric(p=0.22) ~ mean 4.5 steps
                duration = int(self.rng.geometric(p=0.22))
                self.active_honeypots.append(duration)
                if is_mal:
                    reward = +40.0
                    # Containment disruption: 85% probability of resetting attacker to Recon
                    if self.rng.random() < 0.85:
                        self.attacker_stage = 0
                else:
                    reward = -10.0

        elif action == 2:  # ISOLATE
            reward = +15.0 if is_mal else -30.0

        # 3. Check termination
        done = bool(self.current_step >= self.max_steps)

        # 4. Generate next incoming traffic state
        self.current_state_tuple = self._generate_inbound_flow()
        self.current_state_idx = self.encode_state(*self.current_state_tuple)

        info = {
            "step": self.current_step,
            "honeypot_load": len(self.active_honeypots),
            "overflow": overflow,
            "attacker_stage": self.attacker_stage,
            "was_malicious": is_mal,
            "action_taken": action,
            "state_tuple": self.current_state_tuple
        }

        return self.current_state_idx, reward, done, info