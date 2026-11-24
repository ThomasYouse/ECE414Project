# Adaptive Intrusion Detection and Dynamic Honeypot Routing (NetDefendEnv)

This repository contains the standalone, pure-Python simulation environment (`NetDefendEnv`) and tabular reinforcement learning agents (Q-Learning and SARSA) modeling an enterprise network gateway balancing production availability, active decoy defense, and breach containment under physical buffer constraints.

### 1. Requirements & Setup
- Python 3.10+ (Tested on Python 3.11)
- Dependencies: `numpy`, `matplotlib`, `tabulate` (optional for CLI tables)

```powershell
# Clone the repository
git clone [https://github.com/ThomasYouse/ECE414Project.git](https://github.com/ThomasYouse/ECE414Project.git)
cd ECE414Project

# Set up and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

2. Environment Verification & Testing
To run the unit tests and initial simulation scripts:

PowerShell
# Run unit tests
python -m unittest tests/test_env.py

# Run step-by-step gateway trace
python run_demo.py

# Run Milestone 3 training script
python train.py
3. Reproducing Final Experiments & Live Demo (Milestone 4)
To reproduce the Milestone 4 multi-seed evaluation, capacity ablation sweep (C_max in {1, 3, 5, 10}), zero-shot out-of-distribution tests, policy heatmaps, and live side-by-side demonstration:

PowerShell
# Run full Milestone 4 experiment suite and generate plots/metrics
python run_experiments.py

# Run live side-by-side SARSA vs. threshold rule demo
python demo_video.py --steps 20 --delay 0.35