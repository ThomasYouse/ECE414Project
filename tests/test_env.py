import os
import tempfile
import unittest
import numpy as np
from src.environment import NetDefendEnv
from src.agents import TabularQLearningAgent, TabularSARSAAgent


class TestNetDefendEnv(unittest.TestCase):
    def setUp(self):
        self.env = NetDefendEnv(honeypot_capacity=5, max_steps=200, seed=42)

    def test_state_encoding_decoding_bijection(self):
        """Verify that all 96 states map uniquely and reconstruct accurately."""
        for srv in range(4):
            for alt in range(4):
                for frq in range(3):
                    for his in range(2):
                        idx = self.env.encode_state(srv, alt, frq, his)
                        self.assertTrue(0 <= idx < 96)
                        decoded = self.env.decode_state(idx)
                        self.assertEqual(decoded, (srv, alt, frq, his))

    def test_reset_initialization(self):
        """Verify reset re-initializes step count, decoy buffers, and returns valid state index."""
        state_idx, info = self.env.reset(seed=123)
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(len(self.env.active_honeypots), 0)
        self.assertEqual(info["honeypot_load"], 0)
        self.assertIn("is_malicious", info)
        self.assertIn("was_malicious", info)
        self.assertTrue(0 <= state_idx < 96)

    def test_action_bounds(self):
        """Ensure invalid actions raise a ValueError."""
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step(3)
        with self.assertRaises(ValueError):
            self.env.step(-1)

    def test_honeypot_capacity_and_overflow(self):
        """Verify buffer caps at C_max and triggers overflow flag upon saturation."""
        small_env = NetDefendEnv(honeypot_capacity=2, max_steps=50, seed=99)
        small_env.reset()

        overflow_detected = False
        for _ in range(20):
            _, reward, _, info = small_env.step(action=1)
            if info["overflow"]:
                overflow_detected = True
                self.assertIn(reward, [-10.0, -120.0])
                break

        self.assertTrue(overflow_detected, "Buffer overflow should trigger under persistent decoy allocation.")

    def test_episode_termination(self):
        """Verify episode strictly terminates at max_steps."""
        self.env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = self.env.step(action=0)
            steps += 1
        self.assertEqual(steps, 200)
        self.assertTrue(done)

    def test_aggressive_ood_mode(self):
        """Verify out-of-distribution aggressive attacker mode initializes valid stochastic matrix."""
        ood_env = NetDefendEnv(honeypot_capacity=5, attack_mode="aggressive", seed=7)
        row_sums = ood_env.p_att.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, np.ones(3)))
        with self.assertRaises(ValueError):
            NetDefendEnv(attack_mode="invalid_mode")

    def test_agent_update_and_persistence(self):
        """Verify Q-Learning and SARSA updates modify Q-tables and save/load cleanly."""
        q_agent = TabularQLearningAgent(seed=1)
        sarsa_agent = TabularSARSAAgent(seed=1)

        q_before = q_agent.Q[0, 1]
        q_agent.update(0, 1, 40.0, 5, False)
        self.assertNotEqual(q_agent.Q[0, 1], q_before)

        sarsa_before = sarsa_agent.Q[0, 1]
        sarsa_agent.update(0, 1, 40.0, 5, 0, False)
        self.assertNotEqual(sarsa_agent.Q[0, 1], sarsa_before)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_q.npy")
            sarsa_agent.save(path)
            loaded = TabularSARSAAgent()
            loaded.load(path)
            self.assertTrue(np.array_equal(sarsa_agent.Q, loaded.Q))


if __name__ == "__main__":
    unittest.main()