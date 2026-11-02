# Adaptive Intrusion Detection and Dynamic Honeypot Routing (NetDefendEnv)

## Milestone 2: Environment Setup & Verification

This repository contains the standalone, pure-Python simulation environment (`NetDefendEnv`) modeling an enterprise network gateway balancing production availability, active decoy defense, and breach containment under physical buffer constraints.

### 1. Requirements & Setup
- Python 3.10+ (Tested on Python 3.11)
- Dependencies: `numpy`, `tabulate` (optional for CLI tables)

```powershell
# Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/ECE414Project.git
cd ECE414Project

# Set up and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt