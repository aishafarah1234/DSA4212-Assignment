# DSA4212 Bayesian optimisation versus CMA-ES: Multimodal Optimisation
## Project Overview

The study focuses on two characteristics of multimodal landscapes: global basin width and basin separation.
It also examines whether alternative BO acquisition functions and CMA-ES restart strategies can improve global optimisation performance.

## Installation & Setup

**1. Clone the project repository:**

```bash
git clone https://github.com/<your-github-username>/DSA4212-Assignment.git
```

**2. Navigate to the project directory:**

```bash
cd DSA4212-Assignment
```
**3. Create a virtual environment:**
```bash
python -m venv .venv
```
**4. Activate it on Windows powershell:**
```bash
.venv\Scripts\Activate.ps1
```
or on macOS/Linux:
```bash
source .venv/bin/activate
```

**5. Install dependencies**
```bash
python -m pip install -r requirements.txt
```
The main libraries used are NumPy, Pandas, SciPy, Matplotlib, PyTorch, BoTorch, GPyTorch and CMA-ES (cma)



## Running the Experiments

Run the following commands from the repository root with the virtual environment activated.
#### Experiment 1 - Width:
```bash
insert command
```
#### Experiment 3 - Width: BO Acquisition Functions:
```bash
insert command
```
#### Experiment 4 - Width: CMA-ES Restart Strategies:
```bash
insert command
```
#### Experiment 2 - Separation:
```bash
python code/experiment2.py --runs 30 --budget 200
```
#### Experiment 3 - Separation: BO Acquisition Functions:
```bash
python code/experiment3_separation.py --runs 30 --budget 200
```
#### Experiment 4 - Separation: CMA-ES Restart Strategies:
```bash
python code/experiment4_separation.py --runs 30 --budget 200 --restart-interval 60
```
For the restart-interval sensitivity analysis:
```bash
python code/experiment4_separation.py --runs 30 --budget 200 --restart-interval 40 --output results/experiment4_interval40
python code/experiment4_separation.py --runs 30 --budget 200 --restart-interval 100 --output results/experiment4_interval100
```
## Evaluation Metrics

The algorithms are evaluated using:

Success rate: Proportion of runs achieving simple regret ≤ 0.01.

Global basin discovery rate: Proportion of runs finding an objective value below 0.

Mean and median final regret: Measures of solution quality.

Convergence curves: Best-so-far regret across objective evaluations.

Evaluations used: Number of actual objective function evaluations.