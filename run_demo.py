import os
import matplotlib.pyplot as plt
import numpy as np
from src.environment import NetDefendEnv


def run_cli_demonstration():
    print("=" * 75)
    print(" NetDefendEnv: Interactive Gateway Simulation Demonstration")
    print("=" * 75)

    env = NetDefendEnv(honeypot_capacity=3, max_steps=10, seed=42)
    state, info = env.reset()

    actions = ["PASS", "HONEYPOT", "ISOLATE"]
    services = ["Web", "Auth", "DB", "Bastion"]
    alerts = ["None", "Low", "Med", "High"]

    print(f"{'Step':<5} | {'Target Srv':<10} | {'Alert':<6} | {'Flow Type':<10} | {'Action':<10} | {'Reward':<7} | {'Decoy Load':<10} | {'Overflow'}")
    print("-" * 75)

    done = False
    step = 0
    while not done:
        srv, alt, frq, his = info["state_tuple"]
        # Cycle through actions for visual verification
        action = step % 3
        next_state, reward, done, info = env.step(action)
        step += 1

        flow_str = "MALICIOUS" if info["was_malicious"] else "Benign"
        print(f"{step:<5} | {services[srv]:<10} | {alerts[alt]:<6} | {flow_str:<10} | {actions[action]:<10} | {reward:<7.1f} | {info['honeypot_load']:<10} | {info['overflow']}")

    print("=" * 75)
    print("Demonstration run finished successfully.\n")


def generate_milestone_figures():
    os.makedirs("results", exist_ok=True)
    env = NetDefendEnv(honeypot_capacity=5, max_steps=100, seed=101)
    env.reset()

    loads = []
    rewards = []
    actions_taken = []

    # Run 100 steps under a heuristic to create a realistic telemetry trace for Figure 3
    for _ in range(100):
        srv, alt, frq, his = env.current_state_tuple
        risk = alt * 3.0 + frq * 2.0 + his * 4.0 + (2.0 if srv in [2, 3] else 0.0)
        action = 2 if risk >= 9.0 else (1 if risk >= 4.0 else 0)

        _, r, _, info = env.step(action)
        loads.append(info["honeypot_load"])
        rewards.append(r)
        actions_taken.append(action)

    # Plot Execution Output for Figure 3
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    steps = range(1, 101)

    ax1.plot(steps, loads, color="darkorange", lw=2, label="Active Decoy Sessions")
    ax1.axhline(y=5, color="red", linestyle="--", label="Buffer Capacity limit (C_max = 5)")
    ax1.set_ylabel("Honeypot Load")
    ax1.set_title("Figure 3: NetDefendEnv Dynamic Telemetry Trace (100 Decision Steps)", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(steps, np.cumsum(rewards), color="forestgreen", lw=2, label="Cumulative Return")
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Cumulative Reward")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plot_path = os.path.join("results", "figure3_execution_trace.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Figure 3 generated and saved to: {plot_path}")


if __name__ == "__main__":
    run_cli_demonstration()
    generate_milestone_figures()