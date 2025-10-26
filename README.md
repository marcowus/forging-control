# forging-control

## Overview
This project studies data-driven predictive control for an open-die forging
press. The hydraulic press is modeled as a nonlinear continuous-time system with
five states (die displacement, die velocity, two chamber pressures, and
servo-valve spool position) and a single control input (servo-valve tension).
The governing equations, actuator saturation, and forging force calculation are
encoded in `template_model.py`, while Model Predictive Control (MPC) policies
are implemented through do-mpc templates. The repository provides end-to-end
scripts to

1. simulate the nonlinear MPC controller to generate datasets, and
2. train neural surrogates (supervised or unsupervised) that reproduce the MPC
   behavior and can be deployed in closed-loop simulations.

The code is designed around reproducible experiments: each script configures
logging, produces plots of key signals, and saves artifacts (trained models,
metrics, and diagnostic tables) to dedicated `results/` folders.

## Repository layout

```text
Data/                   # Pre-generated closed-loop MPC trajectories (pickle files)
Generate Data/          # Stand-alone MPC simulation to synthesize new datasets
Supervised Learning/    # Feedforward NN controller trained on MPC trajectories
Unsupervised Learning/  # LSTM surrogate model + NN controller with MPC loss
LICENSE
README.md
```

Each learning directory (`Supervised Learning/` and `Unsupervised Learning/`)
contains:

- `Main.py`: orchestrates training, evaluation, and (optionally) closed-loop MPC
  rollouts.
- `Functions.py`: helper utilities for dataset preparation, PyTorch models,
  metrics, plotting, feasibility recovery, and tabulated reports.
- `template_model.py`, `template_mpc.py`, and `template_simulator.py`: do-mpc
  templates shared with the data-generation workflow, defining the plant model,
  controller tuning (10-step horizon, 1 ms sampling, quadratic speed-tracking
  objective, optional pressure constraints), and simulator configuration.
- `results/` and (for supervised learning) `Tables/`: persistent storage for
  trained networks, scalers, metrics, and HTML/CSV summaries.

## Dependencies

All scripts are standard Python programs and expect the following packages:

- Numerical stack: `numpy`, `pandas`, `scikit-learn` (model selection & metrics),
  `scipy` (implicit via dependencies).
- Optimal control: `casadi`, `do-mpc`, and an IPOPT installation (the controller
  template references the MA27 linear solver and a compiled `libcoinhsl.so`).
- Machine learning: `torch` (including CUDA support when available).
- Visualization & reporting: `plotly`, `tabulate`, `alive-progress`.
- Utilities: `notifypy` (desktop notifications), `logging`.

A convenient way to install the requirements is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn torch casadi do-mpc plotly tabulate \
            alive-progress notifypy
```

> **Note:** If your IPOPT or HSL libraries live in a different location, edit
> `template_mpc.py` accordingly (`mpc.settings.nlpsol_opts['ipopt.hsllib']`). 

## Useful resources

- [do-mpc documentation](https://www.do-mpc.com/en/latest)
- [CasADi documentation](https://web.casadi.org/docs)
- [HSL MA27 download request](https://www.hsl.rl.ac.uk/download/MA27/1.0.0/a)

  > The MUMPS linear solver bundled with IPOPT is an open alternative that
  > requires no download request, but expect noticeably slower solve times than
  > MA27 for the MPC problem and feasibility recovery strategy.

## Data

The `Data/` directory stores MPC trajectory datasets collected with different
prediction horizons (`N = 5, 10, 15, 20, 25`). Files follow the naming pattern
`forging_mult_traj_process_noise_N_<horizon>.pkl` and contain serialized do-mpc
results (states, control inputs, time-varying parameters, and solver metadata).
`Generate Data/Main.py` can be used to produce additional datasets with custom
noise levels, initial conditions, or trajectory counts.

### Generating new data

1. Adjust user settings in `Generate Data/Main.py`, such as the number of
   trajectories (`N_TRAJ`), simulation length (`T_TRAJ`), process/measurement
   noise, and IPOPT verbosity.
2. Run the script:
   ```bash
   cd "Generate Data"
   python Main.py
   ```
3. The script prints runtime statistics, produces interactive Plotly figures for
   key signals (state evolution, references, optimization diagnostics), and
   stores the results via `do_mpc.data.save_results`.

## Supervised learning workflow

`Supervised Learning/Main.py` trains a feedforward neural network controller to
replicate the MPC policy using supervised learning:

1. **Configuration.** Toggle flags at the top of the file to enable/disable
   training, MPC rollouts, and feasibility recovery. Set hyperparameters such as
   batch size, epochs, and learning rate.
2. **Data preparation.** The script loads the selected MPC dataset (matching the
   controller horizon), extracts features (`y_dot`, `z`, `ref`) and targets (`u`),
   splits them into train/validation/test partitions, and scales them using the
   utilities in `Functions.py`.
3. **Training.** The `FNNModel` (single hidden layer, 50 neurons by default)
   trains on PyTorch using AdamW and L1 loss. Progress bars, validation metrics,
   and GPU utilization (when available) are logged.
4. **Evaluation & deployment.** After training, the controller can be deployed in
   closed-loop simulations alongside the baseline MPC. Results—tracking error
   (MAE/RMSE/R²), command profiles, and execution time statistics—are stored in
   `results/` and summarized in `Tables/`.

Run the experiment with:

```bash
cd "Supervised Learning"
python Main.py
```

## Unsupervised (hybrid) learning workflow

`Unsupervised Learning/Main.py` augments the supervised controller with a
learned recurrent model and MPC-inspired loss terms:

1. **Model surrogate.** An `LSTMModel` approximates the plant dynamics using
   recurrent features (`y_dot`, `p1`, `p2`, `z`, `u`). Trained parameters and
   feature scalers are loaded from `Model_NN/results/`.
2. **Controller training.** A neural controller is optimized against a composite
   loss (`MPCLoss`) that blends tracking objectives with penalties derived from
   the surrogate dynamics. This supports constraint handling and feasibility
   recovery.
3. **Closed-loop comparison.** Similar to the supervised workflow, the script can
   run MPC and NN controllers in simulation, log metrics, and generate Plotly
   visualizations.

Execute with:

```bash
cd "Unsupervised Learning"
python Main.py
```

## Logging, notifications, and outputs

- Every script configures Python’s `logging` module to write concise logs to both
  the console and `my_log.log` in the working directory.
- Desktop notifications are sent on completion via `notifypy` (disable or remove
  if unsupported on your platform).
- Interactive figures open in your default browser through custom Plotly
  renderers. Saved artifacts (PyTorch checkpoints, scaler pickles, HTML plots)
  reside in the corresponding `results/` subdirectories.

