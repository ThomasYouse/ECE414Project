import os
import matplotlib.pyplot as plt
import numpy as np
from src.environment import NetDefendEnv
from src.agents import (
    TabularQLearningAgent,
    TabularSARSAAgent,
    ThresholdRuleBaseline,
    RandomBaseline,
)


def evaluate_policy(env: NetDefendEnv, agent, is_baseline: bool = False, baseline_type: str = None, num_eval_episodes: int = 50) -> dict:
    """Evaluates a policy greedily across multiple episodes."""
    returns = []
    breaches = 0
    false_positives = 0
    total_flows = 0

    for ep in range(num_eval_episodes):
        state, info = env.reset(seed=1000 + ep)
        done = False
        ep_return = 0.0

        while not done:
            total_flows += 1
            if is_baseline:
                if baseline_type == "threshold":
                    action = ThresholdRuleBaseline.select_action(info["state_tuple"])
                else:
                    action = RandomBaseline.select_action()
            else:
                action = agent.select_action(state, greedy=True)

            next_state, reward, done, next_info = env.step(action)
            ep_return += reward

            # Robust check across reset() and step() dictionary keys
            is_mal = next_info.get("was_malicious", next_info.get("is_malicious", False))

            # Track security metrics
            if is_mal and action == 0:
                breaches += 1
            if not is_mal and action == 2:
                false_positives += 1

            state = next_state
            info = next_info

        returns.append(ep_return)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "breach_count": breaches,
        "fp_count": false_positives,
    }


def train_and_evaluate(episodes: int = 2500, eval_interval: int = 50):
    os.makedirs("results", exist_ok=True)
    env = NetDefendEnv(honeypot_capacity=5, max_steps=200, seed=42)
    eval_env = NetDefendEnv(honeypot_capacity=5, max_steps=200, seed=999)

    q_agent = TabularQLearningAgent(num_states=96, num_actions=3, lr=0.1, gamma=0.95)
    sarsa_agent = TabularSARSAAgent(num_states=96, num_actions=3, lr=0.1, gamma=0.95)

    q_eval_history = []
    sarsa_eval_history = []
    eval_steps = []

    print("Starting Training Experiment...")

    # Training Loop for Q-Learning
    print("Training Tabular Q-Learning...")
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = q_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            q_agent.update(state, action, reward, next_state, done)
            state = next_state
        q_agent.decay_epsilon()

        if (ep + 1) % eval_interval == 0:
            eval_res = evaluate_policy(eval_env, q_agent)
            q_eval_history.append(eval_res["mean_return"])
            eval_steps.append(ep + 1)

    # Training Loop for SARSA
    print("Training Tabular SARSA...")
    for ep in range(episodes):
        state, _ = env.reset()
        action = sarsa_agent.select_action(state)
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = sarsa_agent.select_action(next_state)
            sarsa_agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
        sarsa_agent.decay_epsilon()

        if (ep + 1) % eval_interval == 0:
            eval_res = evaluate_policy(eval_env, sarsa_agent)
            sarsa_eval_history.append(eval_res["mean_return"])

    # Evaluate Baselines
    print("Evaluating Baselines...")
    threshold_metrics = evaluate_policy(eval_env, None, is_baseline=True, baseline_type="threshold")
    random_metrics = evaluate_policy(eval_env, None, is_baseline=True, baseline_type="random")
    final_q_metrics = evaluate_policy(eval_env, q_agent)
    final_sarsa_metrics = evaluate_policy(eval_env, sarsa_agent)

    print("\n" + "=" * 65)
    print(f"{'Method':<20} | {'Mean Return':<12} | {'Breaches':<10} | {'False Drops'}")
    print("-" * 65)
    print(f"{'Q-Learning':<20} | {final_q_metrics['mean_return']:<12.1f} | {final_q_metrics['breach_count']:<10} | {final_q_metrics['fp_count']}")
    print(f"{'SARSA':<20} | {final_sarsa_metrics['mean_return']:<12.1f} | {final_sarsa_metrics['breach_count']:<10} | {final_sarsa_metrics['fp_count']}")
    print(f"{'Threshold Heuristic':<20} | {threshold_metrics['mean_return']:<12.1f} | {threshold_metrics['breach_count']:<10} | {threshold_metrics['fp_count']}")
    print(f"{'Random Agent':<20} | {random_metrics['mean_return']:<12.1f} | {random_metrics['breach_count']:<10} | {random_metrics['fp_count']}")
    print("=" * 65 + "\n")

    # Generate Figure 3: Training Curves
    plt.figure(figsize=(8, 5))
    plt.plot(eval_steps, q_eval_history, label="Tabular Q-Learning", color="royalblue", lw=2)
    plt.plot(eval_steps, sarsa_eval_history, label="Tabular SARSA", color="forestgreen", lw=2)
    plt.axhline(y=threshold_metrics["mean_return"], color="darkorange", linestyle="--", label=f"Threshold Baseline ({threshold_metrics['mean_return']:.1f})")
    plt.axhline(y=random_metrics["mean_return"], color="crimson", linestyle=":", label=f"Random Baseline ({random_metrics['mean_return']:.1f})")
    plt.xlabel("Training Episode", fontsize=11)
    plt.ylabel("Evaluation Mean Return (50 Episodes)", fontsize=11)
    plt.title("Figure 3: Agent Learning Curve vs. Baselines Across 2,500 Episodes", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3_path = os.path.join("results", "figure3_training_curve.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3 to: {fig3_path}")

    # Generate Figure 4: Baseline Comparison Bar Chart
    methods = ["Random", "Threshold Rule", "SARSA", "Q-Learning"]
    returns = [
        random_metrics["mean_return"],
        threshold_metrics["mean_return"],
        final_sarsa_metrics["mean_return"],
        final_q_metrics["mean_return"],
    ]
    colors = ["crimson", "darkorange", "forestgreen", "royalblue"]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(methods, returns, color=colors, width=0.55)
    plt.ylabel("Mean Episodic Return", fontsize=11)
    plt.title("Figure 4: Comparative Evaluation Against Baselines", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (20 if yval >= 0 else -60), f"{yval:.1f}", ha="center", va="bottom" if yval < 0 else "top", fontweight="bold")
    plt.tight_layout()
    fig4_path = os.path.join("results", "figure4_baseline_comparison.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to: {fig4_path}")


if __name__ == "__main__":
    train_and_evaluate()