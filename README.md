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

## Experimental Setup
Experiments 1–5 use a budget of 300 objective evaluations per run. Experiment 6 uses a fixed-generation budget of λ × G per arm (λ = 6, 12, 24 at G = 70, i.e. 420, 840, and 1680 evaluations).

The baseline BO implementation uses a Gaussian Process surrogate with Log Expected Improvement, 10 scrambled Sobol initial observations, input normalisation and manual output standardisation.

The baseline CMA-ES uses a uniformly random initial mean, initial step size 2.0 and no scheduled restarts.

Experiments 1–3 and 5 use 30 runs per configuration. Experiment 4 uses 100 runs per configuration. Experiment 6 uses 30 runs per configuration. The additional Experiment 3 diagnostic uses 10 runs.

## Running the Experiments

Run the following commands from the repository root with the virtual environment activated.
#### Experiment 1 - Effect of global basin width:
Compare baseline BO and CMA-ES while varying global basin curvature

Experiment 1 was implemented as a Jupyter Notebook (code/experiment1.ipynb).

To reproduce the experiment:

1. Activate the project's virtual environment.
2. Launch Jupyter Notebook from the project root:
```bash
jupyter notebook
```
3. Open code/experiment1.ipynb.
4. Select the Python kernel associated with the project's virtual environment.
5. Select Kernel → Restart Kernel and Run All Cells (or the equivalent Run All command).

The notebook executes the baseline BO and CMA-ES experiments and generates the corresponding results and visualisations.

Experiments 2–6 are implemented as Python scripts and can be executed using the commands provided in the following sections from the repository root


#### Experiment 2 - Effect of basin separation:
Compare baseline BO and CMA-ES while varying separation.
```bash
python code/experiment2.py --setup pilot --budget 300 --runs 30 --workers 3 --output results/experiment2
```

Supplementary layout sweep, which the report's §4.2 uses to separate the effect of separation from the effect of the global centre's position:
```bash
python code/experiment2_position.py --layout fixed_midpoint --setup pilot --budget 300 --runs 30 --workers 3 --output results/experiment2_fixed_midpoint
python code/experiment2_position.py --layout fixed_global --setup pilot --budget 300 --runs 30 --workers 3 --output results/experiment2_fixed_global
```

#### Experiment 3 - BO Acquisition:
Compare LogEI, LogPI, UCB-0.5 and UCB-4, with Random and Sobol as reference methods.
```bash
python code/experiment3.py --setup pilot --budget 300 --runs 30 --workers 3 --output results/experiment3
```
sensitivity analysis:
```bash
python code/experiment3.py --cells 5:4 --acquisitions LogEI UCB-4 UCB-64 UCB-256 --setup pilot --budget 300 --runs 10 --workers 3 --output results/experiment3_diagnostic
```

#### Experiment 4 - CMA ES Restart strategies:

Experiment 4 investigates whether scheduled restart strategies improve CMA-ES performance on multimodal landscapes.

Three CMA-ES configurations are compared:

1. CMA-ES: Standard CMA-ES without restarts.

2. CMA-ES-RR: Scheduled random restarts with a fixed population size.

3. CMA-ES-IPOP: Scheduled IPOP-style restarts with population size doubling after each restart.

Random and Sobol sampling are included as reference methods.

Run Experiment 4 with a 60-evaluation restart interval:
```bash
python code/experiment4.py --runs 100 --budget 300 --restart-interval 60 --workers 3 --output results/experiment4_interval60
```
Run Experiment 4 with a 100-evaluation restart interval:
```bash
python code/experiment4.py --runs 100 --budget 300 --restart-interval 100 --workers 3 --output results/experiment4_interval100
```

#### Experiment 5 - Matched initialisation:
Given CMA-ES the same first 10 Sobol evaluations as the saved LogEI runs from Experiment 3 (same seed, same construction, point-by-point verified), charges them to its 300-evaluation budget, and starts CMA-ES from their argmin (CMA-ES-S10). This isolates the effect of the initial design from the search algorithm.

```bash
python code/matched_init.py --source results/experiment3 --cells 5:4 10:4 20:4 10:1 --seed-start 2000 --runs 30 --output results/matched_init
```

Note: matched_init.py imports experiment4.py and calls e4.cma_run for its standard CMA-ES arm, so experiment4.py must be the version from this repository. Run Experiment 4 first if you have not already.


#### Experiment 6 - Fixed-generation population control:

Three single-run CMA-ES arms with populations λ ∈ {6, 12, 24}, each targeting the same total number of generations G = 70. Because each arm completes the same number of generations in a single run, the arms differ only in population width and total evaluations. Note that pycma's tolerance criteria fire earlier for larger populations, so completed generations differ modestly across arms (mean 70.0, 58.5, 47.5 for λ = 6, 12, 24); the contrast is at fixed targeted generations, and the observed population effect on discovery is therefore a lower bound.

```bash
python code/experiment4_fixedgen.py --workers 3
```
Outputs go to results/experiment4_fixedgen/ (summary.csv, paired_tests.csv, individual_runs.csv). --aggregate-only rebuilds them from saved runs.

## Evaluation Metrics

The algorithms are evaluated using:

Branch visitation: Whether an evaluation enters the region governed by the global quadratic branch.

Deep discovery: Whether any evaluation achieves (f(x)<0).

Strict success: Whether final simple regret is at most 0.01.

Final simple regret: Difference between the best observed objective value and the known global minimum.

Grid coverage: Proportion of spatial grid cells visited.

Runtime: Execution time per run.

Restart count: Number of restarts performed in Experiment 4.

For the CMA-ES arms, per-run records also include evaluations_used, evaluation_padding, population_sizes, segment_evaluations and generations_per_run. The summary.csv for Experiment 4 reports mean_restart_count, n_schedule_clipped and frac_schedule_clipped; Experiment 6's summary.csv reports popsize, generations_target, generations_completed_mean and budget.

The main statistical analysis uses Wilson confidence intervals and exact paired McNemar tests with Holm correction.

## Validation and Reproducibility

Two additional checks are included to support experimental reproducibility.

### Pilot Configuration Check

A small validation run was conducted to verify the shared experimental configuration used in Experiments 1–3.

To reproduce the check:

```bash
python code/experiment2.py --setup pilot --budget 300 --cells 10:4 --acquisitions LogEI CMA-ES --seed-start 0 --runs 10 --output results/check_pilot --workers 3
```

The check evaluates BO-LogEI and CMA-ES over 10 seeds each, using 300 objective evaluations on the landscape \(\alpha=10,D=4\).

Results are stored in `results/check_pilot/`.

### Experiment 3 Reproducibility Check

The reproducibility script compares overlapping runs from the main Experiment 3 and the UCB sensitivity diagnostic.

Run:

```bash
python code/check_reproducibility.py
```

The script compares runs matched by landscape, acquisition function and seed, and reports differences in best objective values and global discovery outcomes.

The check requires both Experiment 3 result directories to exist and to contain overlapping seeds.

For matching deterministic runs, the expected maximum difference in best objective value is zero or within numerical tolerance, with no differences in discovery outcomes.