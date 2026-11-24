import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.environment import NetDefendEnv
from src.agents import (
    TabularQLearningAgent,
    TabularSARSAAgent,
    ThresholdRuleBaseline,
    RandomBaseline,
)


def evaluate_policy(
    env: NetDefendEnv,
    agent=None,
    baseline_type: str = None,
    num_eval_episodes: int = 50,
    eval_seed_base: int = 5000,
) -> dict:
    """Evaluates an RL agent or baseline across fixed evaluation seeds."""
    returns = []
    breaches = 0
    false_positives = 0
    overflows = 0
    captures = 0

    rand_baseline = RandomBaseline(seed=eval_seed_base) if baseline_type == "random" else None

    for ep in range(num_eval_episodes):
        state, info = env.reset(seed=eval_seed_base + ep)
        done = False
        ep_return = 0.0

        while not done:
            if baseline_type == "threshold":
                action = ThresholdRuleBaseline.select_action(info["state_tuple"])
            elif baseline_type == "random":
                action = rand_baseline.select_action()
            else:
                action = agent.select_action(state, greedy=True)

            next_state, reward, done, next_info = env.step(action)
            ep_return += reward

            is_mal = next_info["was_malicious"]
            if is_mal and action == 0:
                breaches += 1
            if not is_mal and action == 2:
                false_positives += 1
            if next_info["overflow"]:
                overflows += 1
            if is_mal and action == 1 and not next_info["overflow"]:
                captures += 1

            state = next_state
            info = next_info

        returns.append(ep_return)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "breaches_per_ep": breaches / num_eval_episodes,
        "fp_per_ep": false_positives / num_eval_episodes,
        "overflows_per_ep": overflows / num_eval_episodes,
        "captures_per_ep": captures / num_eval_episodes,
    }


def train_agent(
    algo: str,
    honeypot_capacity: int = 5,
    episodes: int = 1500,
    seed: int = 0,
):
    """Trains a single Q-Learning or SARSA agent and returns (agent, training_returns)."""
    env = NetDefendEnv(honeypot_capacity=honeypot_capacity, max_steps=200, seed=seed)

    if algo == "q_learning":
        agent = TabularQLearningAgent(
            num_states=96, num_actions=3,
            lr=0.1, lr_min=0.005, lr_decay=0.998,
            gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9965,
            q_init=10.0, seed=seed,
        )
    elif algo == "sarsa":
        agent = TabularSARSAAgent(
            num_states=96, num_actions=3,
            lr=0.1, lr_min=0.005, lr_decay=0.998,
            gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9965,
            q_init=10.0, seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    train_returns = np.zeros(episodes, dtype=np.float64)

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0.0

        if algo == "q_learning":
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                ep_return += reward
                state = next_state
        else:
            action = agent.select_action(state)
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                ep_return += reward
                state = next_state
                action = next_action

        agent.decay_parameters()
        train_returns[ep] = ep_return

    return agent, train_returns


def moving_average(data: np.ndarray, window: int = 30) -> np.ndarray:
    """Computes trailing rolling average along the last axis."""
    kernel = np.ones(window) / window
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), axis=-1, arr=data)


def run_all_experiments():
    t0 = time.time()
    os.makedirs("results", exist_ok=True)
    summary_lines = []

    def log(msg: str = ""):
        print(msg)
        summary_lines.append(msg)

    log("=" * 78)
    log(" MILESTONE 4: FINAL EXPERIMENTAL SUITE (NetDefendEnv)")
    log("=" * 78)

    # =========================================================================
    # EXPERIMENT 1 & 2: Multi-Seed Training & Baseline Comparison (C_max = 5)
    # =========================================================================
    num_seeds = 5
    episodes = 1500
    eval_env_std = NetDefendEnv(honeypot_capacity=5, max_steps=200, benign_ratio=0.70, attack_mode="standard")

    log(f"\n[Experiment 1] Training Q-Learning and SARSA across {num_seeds} seeds ({episodes} episodes each)...")

    q_curves, sarsa_curves = [], []
    q_agents, sarsa_agents = [], []
    q_metrics_list, sarsa_metrics_list = [], []

    for seed in range(num_seeds):
        q_ag, q_ret = train_agent("q_learning", honeypot_capacity=5, episodes=episodes, seed=seed)
        s_ag, s_ret = train_agent("sarsa", honeypot_capacity=5, episodes=episodes, seed=seed)

        q_curves.append(q_ret)
        sarsa_curves.append(s_ret)
        q_agents.append(q_ag)
        sarsa_agents.append(s_ag)

        q_metrics_list.append(evaluate_policy(eval_env_std, agent=q_ag, num_eval_episodes=50))
        sarsa_metrics_list.append(evaluate_policy(eval_env_std, agent=s_ag, num_eval_episodes=50))
        log(f"  Seed {seed} done | Q-Learning: {q_metrics_list[-1]['mean_return']:7.1f} | SARSA: {sarsa_metrics_list[-1]['mean_return']:7.1f}")

    # Save best-performing Q-tables for video demonstration and inspection
    best_q_idx = int(np.argmax([m["mean_return"] for m in q_metrics_list]))
    best_s_idx = int(np.argmax([m["mean_return"] for m in sarsa_metrics_list]))
    q_agents[best_q_idx].save(os.path.join("results", "q_learning_q_table.npy"))
    sarsa_agents[best_s_idx].save(os.path.join("results", "sarsa_q_table.npy"))

    # Evaluate baselines on standard environment
    rand_metrics = evaluate_policy(eval_env_std, baseline_type="random", num_eval_episodes=50)
    thresh_metrics = evaluate_policy(eval_env_std, baseline_type="threshold", num_eval_episodes=50)

    def aggregate(m_list):
        keys = ["mean_return", "breaches_per_ep", "fp_per_ep", "overflows_per_ep", "captures_per_ep"]
        return {
            k: (float(np.mean([m[k] for m in m_list])), float(np.std([m[k] for m in m_list])))
            for k in keys
        }

    q_agg = aggregate(q_metrics_list)
    s_agg = aggregate(sarsa_metrics_list)

    log("\n[Experiment 2] Multi-Seed Evaluation Summary (C_max = 5, 50 Eval Eps / Seed)")
    log("-" * 88)
    log(f"{'Method':<20} | {'Mean Return':<16} | {'Breaches/Ep':<12} | {'False Drops/Ep':<14} | {'Overflows/Ep'}")
    log("-" * 88)
    log(f"{'Random Baseline':<20} | {rand_metrics['mean_return']:7.1f} +/- {rand_metrics['std_return']:4.1f} | {rand_metrics['breaches_per_ep']:<12.2f} | {rand_metrics['fp_per_ep']:<14.2f} | {rand_metrics['overflows_per_ep']:.2f}")
    log(f"{'Threshold Heuristic':<20} | {thresh_metrics['mean_return']:7.1f} +/- {thresh_metrics['std_return']:4.1f} | {thresh_metrics['breaches_per_ep']:<12.2f} | {thresh_metrics['fp_per_ep']:<14.2f} | {thresh_metrics['overflows_per_ep']:.2f}")
    log(f"{'Tabular Q-Learning':<20} | {q_agg['mean_return'][0]:7.1f} +/- {q_agg['mean_return'][1]:4.1f} | {q_agg['breaches_per_ep'][0]:<12.2f} | {q_agg['fp_per_ep'][0]:<14.2f} | {q_agg['overflows_per_ep'][0]:.2f}")
    log(f"{'Tabular SARSA':<20} | {s_agg['mean_return'][0]:7.1f} +/- {s_agg['mean_return'][1]:4.1f} | {s_agg['breaches_per_ep'][0]:<12.2f} | {s_agg['fp_per_ep'][0]:<14.2f} | {s_agg['overflows_per_ep'][0]:.2f}")
    log("-" * 88)

    # -------------------------------------------------------------------------
    # FIGURE 3: Final Multi-Seed Learning Curve (Mean +/- 1 Std)
    # -------------------------------------------------------------------------
    window = 30
    q_smooth = moving_average(np.array(q_curves), window=window)
    s_smooth = moving_average(np.array(sarsa_curves), window=window)
    x_axis = np.arange(window, episodes + 1)

    q_mean, q_std = q_smooth.mean(axis=0), q_smooth.std(axis=0)
    s_mean, s_std = s_smooth.mean(axis=0), s_smooth.std(axis=0)

    plt.figure(figsize=(8.5, 5))
    plt.plot(x_axis, q_mean, label="Tabular Q-Learning (Mean, 5 Seeds)", color="royalblue", lw=2)
    plt.fill_between(x_axis, q_mean - q_std, q_mean + q_std, color="royalblue", alpha=0.18)

    plt.plot(x_axis, s_mean, label="Tabular SARSA (Mean, 5 Seeds)", color="forestgreen", lw=2)
    plt.fill_between(x_axis, s_mean - s_std, s_mean + s_std, color="forestgreen", alpha=0.18)

    plt.axhline(thresh_metrics["mean_return"], color="darkorange", linestyle="--", lw=1.8,
                label=f"Threshold Heuristic ({thresh_metrics['mean_return']:.1f})")
    plt.axhline(rand_metrics["mean_return"], color="crimson", linestyle=":", lw=1.8,
                label=f"Random Baseline ({rand_metrics['mean_return']:.1f})")

    plt.xlabel("Training Episode", fontsize=11)
    plt.ylabel(f"Episodic Return ({window}-Ep Rolling Mean)", fontsize=11)
    plt.title("Figure 3: Multi-Seed Training Convergence vs. Baselines (C_max = 5)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "figure3_final_training_curve.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 4: Comparative Performance & Security Error Breakdown
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    methods = ["Random", "Threshold\nRule", "Q-Learning", "SARSA"]
    ret_means = [rand_metrics["mean_return"], thresh_metrics["mean_return"], q_agg["mean_return"][0], s_agg["mean_return"][0]]
    ret_stds = [rand_metrics["std_return"] / np.sqrt(50), thresh_metrics["std_return"] / np.sqrt(50), q_agg["mean_return"][1], s_agg["mean_return"][1]]
    colors = ["crimson", "darkorange", "royalblue", "forestgreen"]

    bars = ax1.bar(methods, ret_means, yerr=ret_stds, capsize=5, color=colors, width=0.55)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("Mean Episodic Return", fontsize=10.5)
    ax1.set_title("(A) Mean Episodic Return (+/- Std)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, ret_means):
        offset = 90 if val >= 0 else -240
        ax1.text(bar.get_x() + bar.get_width() / 2.0, val + offset, f"{val:.1f}",
                 ha="center", va="bottom" if val >= 0 else "top", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(min(ret_means) - 550, max(ret_means) + 450)

    # Panel B: Security Failure Metrics (excluding Random for clear scaling)
    comp_methods = ["Threshold Rule", "Q-Learning", "SARSA"]
    breaches_vals = [thresh_metrics["breaches_per_ep"], q_agg["breaches_per_ep"][0], s_agg["breaches_per_ep"][0]]
    fp_vals = [thresh_metrics["fp_per_ep"], q_agg["fp_per_ep"][0], s_agg["fp_per_ep"][0]]
    ov_vals = [thresh_metrics["overflows_per_ep"], q_agg["overflows_per_ep"][0], s_agg["overflows_per_ep"][0]]

    x = np.arange(len(comp_methods))
    w = 0.25
    ax2.bar(x - w, breaches_vals, width=w, label="Breaches / Ep", color="#d62728")
    ax2.bar(x, fp_vals, width=w, label="False Drops / Ep", color="#ff7f0e")
    ax2.bar(x + w, ov_vals, width=w, label="Overflows / Ep", color="#9467bd")
    ax2.set_xticks(x)
    ax2.set_xticklabels(comp_methods)
    ax2.set_ylabel("Mean Events per Episode (200 Flows)", fontsize=10.5)
    ax2.set_title("(B) Security Failure Modes (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 4: Final Policy Performance & Security Trade-Off Analysis", fontsize=12.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "figure4_baseline_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # EXPERIMENT 3A: Capacity Ablation Sweep (C_max in {1, 3, 5, 10})
    # =========================================================================
    log("\n[Experiment 3A] Honeypot Buffer Capacity Ablation (C_max in {1, 3, 5, 10}, 3 Seeds Each)...")
    capacities = [1, 3, 5, 10]
    ablation_seeds = 3

    cap_results = {"threshold": [], "q_learning": [], "sarsa": []}
    cap_overflows = {"threshold": [], "q_learning": [], "sarsa": []}

    log("-" * 82)
    log(f"{'C_max':<6} | {'Threshold Ret (Ov)':<22} | {'Q-Learning Ret (Ov)':<22} | {'SARSA Ret (Ov)'}")
    log("-" * 82)

    for cap in capacities:
        ev = NetDefendEnv(honeypot_capacity=cap, max_steps=200, benign_ratio=0.70, attack_mode="standard")
        t_res = evaluate_policy(ev, baseline_type="threshold", num_eval_episodes=50)
        cap_results["threshold"].append(t_res["mean_return"])
        cap_overflows["threshold"].append(t_res["overflows_per_ep"])

        q_r, q_ov, s_r, s_ov = [], [], [], []
        for seed in range(ablation_seeds):
            qa, _ = train_agent("q_learning", honeypot_capacity=cap, episodes=1500, seed=seed)
            sa, _ = train_agent("sarsa", honeypot_capacity=cap, episodes=1500, seed=seed)
            qm = evaluate_policy(ev, agent=qa, num_eval_episodes=50)
            sm = evaluate_policy(ev, agent=sa, num_eval_episodes=50)
            q_r.append(qm["mean_return"])
            q_ov.append(qm["overflows_per_ep"])
            s_r.append(sm["mean_return"])
            s_ov.append(sm["overflows_per_ep"])

        cap_results["q_learning"].append(float(np.mean(q_r)))
        cap_overflows["q_learning"].append(float(np.mean(q_ov)))
        cap_results["sarsa"].append(float(np.mean(s_r)))
        cap_overflows["sarsa"].append(float(np.mean(s_ov)))

        log(
            f"{cap:<6} | {t_res['mean_return']:7.1f} ({t_res['overflows_per_ep']:5.2f} ov)    "
            f"| {np.mean(q_r):7.1f} ({np.mean(q_ov):5.2f} ov)    "
            f"| {np.mean(s_r):7.1f} ({np.mean(s_ov):5.2f} ov)"
        )
    log("-" * 82)

    # =========================================================================
    # EXPERIMENT 3B: Zero-Shot Out-of-Distribution (OOD) Robustness
    # =========================================================================
    log("\n[Experiment 3B] Zero-Shot Out-of-Distribution (OOD) Generalization (Trained at C_max=5)...")
    scenarios = [
        ("Standard\n(30% Mal)", dict(honeypot_capacity=5, benign_ratio=0.70, attack_mode="standard")),
        ("OOD Fast\nEscalation", dict(honeypot_capacity=5, benign_ratio=0.70, attack_mode="aggressive")),
        ("OOD Surge\n(50% Mal)", dict(honeypot_capacity=5, benign_ratio=0.50, attack_mode="aggressive")),
    ]

    ood_returns = {"threshold": [], "q_learning": [], "sarsa": []}
    log("-" * 78)
    log(f"{'Scenario':<25} | {'Threshold':<14} | {'Q-Learning':<14} | {'SARSA'}")
    log("-" * 78)

    for label, cfg in scenarios:
        ev = NetDefendEnv(max_steps=200, **cfg)
        tm = evaluate_policy(ev, baseline_type="threshold", num_eval_episodes=50)["mean_return"]
        qm = float(np.mean([evaluate_policy(ev, agent=a, num_eval_episodes=50)["mean_return"] for a in q_agents]))
        sm = float(np.mean([evaluate_policy(ev, agent=a, num_eval_episodes=50)["mean_return"] for a in sarsa_agents]))

        ood_returns["threshold"].append(tm)
        ood_returns["q_learning"].append(qm)
        ood_returns["sarsa"].append(sm)
        clean_lbl = label.replace("\n", " ")
        log(f"{clean_lbl:<25} | {tm:<14.1f} | {qm:<14.1f} | {sm:.1f}")
    log("-" * 78)

    # -------------------------------------------------------------------------
    # FIGURE 5: Technical Challenges — Capacity Ablation & OOD Robustness
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax1.plot(capacities, cap_results["q_learning"], marker="o", lw=2, color="royalblue", label="Tabular Q-Learning")
    ax1.plot(capacities, cap_results["sarsa"], marker="s", lw=2, color="forestgreen", label="Tabular SARSA")
    ax1.plot(capacities, cap_results["threshold"], marker="^", lw=2, linestyle="--", color="darkorange", label="Threshold Heuristic")
    ax1.set_xticks(capacities)
    ax1.set_xlabel("Honeypot Buffer Capacity (C_max)", fontsize=10.5)
    ax1.set_ylabel("Mean Episodic Return", fontsize=10.5)
    ax1.set_title("(A) Capacity Bottleneck Ablation Study", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    x_ood = np.arange(len(scenarios))
    w = 0.25
    ax2.bar(x_ood - w, ood_returns["threshold"], width=w, color="darkorange", label="Threshold Heuristic")
    ax2.bar(x_ood, ood_returns["q_learning"], width=w, color="royalblue", label="Tabular Q-Learning")
    ax2.bar(x_ood + w, ood_returns["sarsa"], width=w, color="forestgreen", label="Tabular SARSA")
    ax2.set_xticks(x_ood)
    ax2.set_xticklabels([s[0] for s in scenarios], fontsize=9.5)
    ax2.set_ylabel("Mean Episodic Return", fontsize=10.5)
    ax2.set_title("(B) Zero-Shot Out-of-Distribution Robustness", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 5: Buffer Capacity Constraints & Out-of-Distribution Generalization", fontsize=12.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "figure5_ablation_and_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # EXPERIMENT 4: Policy Interpretability Heatmap (96 States -> Actions)
    # =========================================================================
    def build_policy_grid(policy_fn) -> np.ndarray:
        # Shape: (4 services, 24 telemetry profiles = 4 alerts x 3 freqs x 2 histories)
        grid = np.zeros((4, 24), dtype=int)
        for srv in range(4):
            col = 0
            for alt in range(4):
                for frq in range(3):
                    for his in range(2):
                        s_idx = NetDefendEnv.encode_state(srv, alt, frq, his)
                        grid[srv, col] = policy_fn(s_idx, (srv, alt, frq, his))
                        col += 1
        return grid

    grid_thresh = build_policy_grid(lambda idx, tup: ThresholdRuleBaseline.select_action(tup))
    grid_q = build_policy_grid(lambda idx, tup: q_agents[best_q_idx].select_action(idx, greedy=True))
    grid_sarsa = build_policy_grid(lambda idx, tup: sarsa_agents[best_s_idx].select_action(idx, greedy=True))

    cmap = ListedColormap(["#4caf50", "#ff9800", "#e53935"])  # 0: PASS (Green), 1: HONEYPOT (Orange), 2: ISOLATE (Red)
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.2), sharex=True)
    titles = [
        "Static Threshold Heuristic Policy",
        "Learned Tabular Q-Learning Policy (Best Seed, C_max = 5)",
        "Learned Tabular SARSA Policy (Best Seed, C_max = 5)",
    ]
    services = ["Web", "Auth", "DB", "Bastion"]

    for ax, grid, title in zip(axes, [grid_thresh, grid_q, grid_sarsa], titles):
        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=2)
        ax.set_yticks(range(4))
        ax.set_yticklabels(services, fontsize=9.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        for x_sep in [5.5, 11.5, 17.5]:
            ax.axvline(x_sep, color="white", lw=2)

    axes[-1].set_xticks([2.5, 8.5, 14.5, 20.5])
    axes[-1].set_xticklabels(["Alert: None\n(Freq 0-2, Hist 0-1)", "Alert: Low\n(Freq 0-2, Hist 0-1)",
                              "Alert: Medium\n(Freq 0-2, Hist 0-1)", "Alert: High\n(Freq 0-2, Hist 0-1)"], fontsize=9.5)

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025, pad=0.02, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["PASS (0)", "HONEYPOT (1)", "ISOLATE (2)"])
    fig.suptitle("Figure 6: Decision Boundary Comparison Across All 96 Discrete States", fontsize=12.5, fontweight="bold", y=0.99)
    plt.savefig(os.path.join("results", "figure6_policy_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - t0
    log(f"\nAll experiments finished in {elapsed:.1f}s.")
    log("Generated files in results/:")
    log("  - figure3_final_training_curve.png")
    log("  - figure4_baseline_comparison.png")
    log("  - figure5_ablation_and_robustness.png")
    log("  - figure6_policy_heatmap.png")
    log("  - q_learning_q_table.npy & sarsa_q_table.npy")

    with open(os.path.join("results", "milestone4_metrics_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print("  - milestone4_metrics_summary.txt")


if __name__ == "__main__":
    run_all_experiments()