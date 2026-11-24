import os
import time
import argparse
from src.environment import NetDefendEnv
from src.agents import TabularSARSAAgent, ThresholdRuleBaseline


def format_load_bar(load: int, capacity: int) -> str:
    filled = "#" * load
    empty = "-" * (capacity - load)
    return f"[{filled}{empty}] {load}/{capacity}"


def run_live_demo(steps: int = 25, delay: float = 0.35, seed: int = 2026):
    q_path = os.path.join("results", "sarsa_q_table.npy")
    if not os.path.exists(q_path):
        print("Trained Q-table not found. Please run `python run_experiments.py` first.")
        return

    sarsa_agent = TabularSARSAAgent()
    sarsa_agent.load(q_path)

    env_rl = NetDefendEnv(honeypot_capacity=5, max_steps=steps, seed=seed)
    env_base = NetDefendEnv(honeypot_capacity=5, max_steps=steps, seed=seed)

    s_rl, info_rl = env_rl.reset(seed=seed)
    _, info_base = env_base.reset(seed=seed)

    actions = ["PASS", "HONEYPOT", "ISOLATE"]
    services = ["Web", "Auth", "DB", "Bastion"]
    alerts = ["None", "Low", "Med", "High"]

    print("=" * 98)
    print("  NetDefendEnv LIVE DEMO: Trained SARSA Agent vs. Static Threshold Firewall (Identical Traffic)")
    print("=" * 98)
    print(f"{'Step':<4} | {'Flow (Srv/Alert)':<16} | {'Truth':<9} || {'SARSA Act':<9} {'Rew':>6} {'Buffer':<11} || {'Rule Act':<9} {'Rew':>6}")
    print("-" * 98)

    rl_total, base_total = 0.0, 0.0
    rl_breach, base_breach = 0, 0
    rl_fp, base_fp = 0, 0

    for t in range(1, steps + 1):
        srv, alt, frq, his = info_rl["state_tuple"]
        a_rl = sarsa_agent.select_action(s_rl, greedy=True)
        a_base = ThresholdRuleBaseline.select_action(info_base["state_tuple"])

        s_rl, r_rl, _, info_rl = env_rl.step(a_rl)
        _, r_base, _, info_base = env_base.step(a_base)

        rl_total += r_rl
        base_total += r_base

        is_mal = info_rl["was_malicious"]
        if is_mal and a_rl == 0:
            rl_breach += 1
        if is_mal and a_base == 0:
            base_breach += 1
        if not is_mal and a_rl == 2:
            rl_fp += 1
        if not is_mal and a_base == 2:
            base_fp += 1

        truth_str = "MALICIOUS" if is_mal else "Benign"
        flow_desc = f"{services[srv]}/{alerts[alt]}"
        buf_bar = format_load_bar(info_rl["honeypot_load"], env_rl.honeypot_capacity)

        print(
            f"{t:<4} | {flow_desc:<16} | {truth_str:<9} || "
            f"{actions[a_rl]:<9} {r_rl:>6.0f} {buf_bar:<11} || "
            f"{actions[a_base]:<9} {r_base:>6.0f}"
        )
        time.sleep(delay)

    print("=" * 98)
    print(f"  FINAL SCOREBOARD ({steps} Flows):")
    print(f"    Trained SARSA Agent   -> Return: {rl_total:+7.1f} | Breaches: {rl_breach} | False Drops: {rl_fp}")
    print(f"    Threshold Firewall    -> Return: {base_total:+7.1f} | Breaches: {base_breach} | False Drops: {base_fp}")
    print("=" * 98)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live side-by-side demo for Milestone 4 video.")
    parser.add_argument("--steps", type=int, default=25, help="Number of gateway steps to simulate.")
    parser.add_argument("--delay", type=float, default=0.35, help="Seconds to pause between steps.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducible traffic.")
    args = parser.parse_args()
    run_live_demo(steps=args.steps, delay=args.delay, seed=args.seed)