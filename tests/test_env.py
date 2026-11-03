import unittest
from src.environment import NetDefendEnv


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
            _, reward, _, info = small_env.step(action=1)  # Force HONEYPOT action
            if info["overflow"]:
                overflow_detected = True
                self.assertIn(reward, [-10.0, -120.0])
                break

        self.assertTrue(overflow_detected, "Buffer overflow condition should trigger under persistent decoy allocation.")

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


if __name__ == "__main__":
    unittest.main()