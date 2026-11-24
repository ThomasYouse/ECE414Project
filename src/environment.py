import math
import random
import numpy as np


class NetDefendEnv:
    """
    Adaptive Intrusion Detection and Dynamic Honeypot Routing Simulator.
    Models an enterprise entrance gateway protecting 4 production services
    (Web, Auth, Database, Bastion) backed by a pooled honeypot buffer.
    Supports 'standard' and out-of-distribution 'aggressive' attacker dynamics.
    """

    def __init__(
        self,
        honeypot_capacity: int = 5,
        max_steps: int = 200,
        benign_ratio: float = 0.70,
        attack_mode: str = "standard",
        seed: int = None,
    ):
        if attack_mode not in ("standard", "aggressive"):
            raise ValueError(f"Invalid attack_mode '{attack_mode}'. Must be 'standard' or 'aggressive'.")

        self.honeypot_capacity = honeypot_capacity
        self.max_steps = max_steps
        self.benign_ratio = benign_ratio
        self.attack_mode = attack_mode

        # State dimensions: 4 services x 4 alert levels x 3 request frequencies x 2 IP histories = 96 states
        self.num_services = 4
        self.num_alerts = 4
        self.num_frequencies = 3
        self.num_histories = 2
        self.num_states = 96
        self.num_actions = 3  # 0: PASS, 1: HONEYPOT, 2: ISOLATE

        # Attacker hidden progression: Recon (0) -> Lateral (1) -> Exfil (2)
        if attack_mode == "aggressive":
            # Out-of-distribution rapid escalation profile (fast Recon -> Exfil jumps)
            self.p_att = np.array([
                [0.20, 0.45, 0.35],
                [0.05, 0.35, 0.60],
                [0.00, 0.10, 0.90],
            ], dtype=np.float64)
        else:
            # Standard MITRE ATT&CK progression
            self.p_att = np.array([
                [0.60, 0.35, 0.05],
                [0.10, 0.65, 0.25],
                [0.00, 0.20, 0.80],
            ], dtype=np.float64)

        self._p_att_cdf = np.cumsum(self.p_att, axis=1).tolist()
        self._log_1_minus_p = math.log(1.0 - 0.22)
        self.py_rng = random.Random(seed)
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

    @staticmethod
    def _sample_4(u: float, c0: float, c1: float, c2: float) -> int:
        if u < c0:
            return 0
        if u < c1:
            return 1
        if u < c2:
            return 2
        return 3

    @staticmethod
    def _sample_3(u: float, c0: float, c1: float) -> int:
        if u < c0:
            return 0
        if u < c1:
            return 1
        return 2

    def reset(self, seed: int = None) -> tuple[int, dict]:
        """Resets the environment to initial conditions."""
        if seed is not None:
            self.py_rng = random.Random(seed)

        self.current_step = 0
        self.active_honeypots = []  # Tracks remaining step durations for allocated decoys
        self.attacker_stage = 0     # 0: Recon, 1: Lateral Movement, 2: Exfiltration
        self.current_flow_is_malicious = False

        self.current_state_tuple = self._generate_inbound_flow()
        self.current_state_idx = self.encode_state(*self.current_state_tuple)

        info = {
            "step": self.current_step,
            "honeypot_load": 0,
            "overflow": False,
            "attacker_stage": self.attacker_stage,
            "is_malicious": self.current_flow_is_malicious,
            "was_malicious": self.current_flow_is_malicious,
            "state_tuple": self.current_state_tuple,
        }
        return self.current_state_idx, info

    def _generate_inbound_flow(self) -> tuple[int, int, int, int]:
        """Emits observable flow metadata conditioned on traffic type and attacker stage."""
        rand = self.py_rng.random
        self.current_flow_is_malicious = bool(rand() > self.benign_ratio)

        if not self.current_flow_is_malicious:
            # Benign distribution: heavily Web/Auth, low alerts, clean IP history
            service = self._sample_4(rand(), 0.55, 0.85, 0.95)
            alert = self._sample_4(rand(), 0.75, 0.93, 0.99)
            freq = self._sample_3(rand(), 0.80, 0.97)
            history = 0 if rand() < 0.96 else 1
        else:
            # Progress attacker's hidden Markov chain
            cdf = self._p_att_cdf[self.attacker_stage]
            self.attacker_stage = self._sample_3(rand(), cdf[0], cdf[1])

            if self.attacker_stage == 0:  # Reconnaissance
                service = self._sample_4(rand(), 0.40, 0.70, 0.80)
                alert = self._sample_4(rand(), 0.20, 0.70, 0.95)
                freq = self._sample_3(rand(), 0.30, 0.80)
                history = 0 if rand() < 0.70 else 1
            elif self.attacker_stage == 1:  # Lateral Movement
                service = self._sample_4(rand(), 0.10, 0.45, 0.80)
                alert = self._sample_4(rand(), 0.05, 0.30, 0.80)
                freq = self._sample_3(rand(), 0.20, 0.80)
                history = 0 if rand() < 0.40 else 1
            else:  # Exfiltration
                service = self._sample_4(rand(), 0.05, 0.10, 0.85)
                alert = self._sample_4(rand(), 0.02, 0.10, 0.45)
                freq = self._sample_3(rand(), 0.10, 0.40)
                history = 0 if rand() < 0.15 else 1

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
                # Exact Geometric(p=0.22) sample (mean ~4.55 steps) via inverse transform
                u = self.py_rng.random()
                duration = int(math.log(max(1e-12, 1.0 - u)) / self._log_1_minus_p) + 1
                self.active_honeypots.append(duration)
                if is_mal:
                    reward = +40.0
                    # Containment disruption: 85% probability of resetting attacker to Recon
                    if self.py_rng.random() < 0.85:
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
            "is_malicious": self.current_flow_is_malicious,
            "was_malicious": is_mal,
            "action_taken": action,
            "state_tuple": self.current_state_tuple,
        }

        return self.current_state_idx, reward, done, info