# Payload Transport Research

## Project Overview

This Python research prototype simulates multiple UAVs transporting a shared payload in the X–Z plane. It combines payload dynamics, a PD controller with feedforward, dynamic control allocation, and reference trajectory generation. It logs each run for plotting and exports JSON replays for a separate public GitHub Pages viewer.

## Repository Structure

```text
.
├── run_simulation.py                  # Simulation entry point
├── config.yaml                        # Model, controller, and run settings
├── requirements.txt
├── controllers/
│   ├── controller_block.py            # PD controller with feedforward
│   └── control_allocation_block.py    # Dynamic force allocation
├── models/
│   ├── plant_block.py                 # Payload dynamics
│   └── uav_block.py                   # Attachment and UAV positions
├── trajectories/
│   └── reference_trajectory_block.py
├── visualization/
│   ├── simulation_logger.py           # Run histories and plots
│   ├── export.py                      # JSON replay writer
│   ├── publish.py                     # Public replay and manifest updates
│   └── index.html                     # Browser replay viewer
└── tests/
    └── test_publish.py
```

## Initial Setup

With Git and Python installed, run these commands in PowerShell. Create a local `.venv` on each computer.

```powershell
git clone https://github.com/arloe-rgb/payload-transport-research.git
Set-Location .\payload-transport-research
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the `py` launcher is unavailable, use `python -m venv .venv` to create the environment. If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` in the installation and run commands.

## Running a Simulation

With the environment active, run this from the research repository root.

```powershell
python .\run_simulation.py
```

The script reads `config.yaml` for the payload, controller, allocator, timing, and reference trajectory settings. After the run, Matplotlib shows payload tracking, the X–Z path, UAV forces, and allocator diagnostics. Close the plots to see the final payload state in the terminal.

## Exporting Simulation Results

Run the simulation in export mode to save a JSON replay without opening plots.

```powershell
python .\run_simulation.py --export-replay .\results\trial.json
```

The script saves the replay to `results/trial.json` in the research checkout and creates the output directory if needed. Change the path after `--export-replay` to choose a different filename.

The JSON records timestamps, payload and reference states, allocated forces, allocator diagnostics, and world-frame X–Z positions for the payload attachments and UAVs. UAV positions are reconstructed from the force directions and configured cable lengths. At zero force, the corresponding UAV position is recorded as `[null, null]`.

The replay stays local until you publish it. To preview it locally, open `visualization/index.html` in a browser and choose **Open replay** to select the JSON file.

## Publishing a Simulation to GitHub Pages

Run simulations and export JSON in the research checkout. Publish selected replays from the separate visualization checkout.

- Research simulation repository — [payload-transport-research](https://github.com/arloe-rgb/payload-transport-research)
- Public visualization repository — [payload-transport-visualization](https://github.com/arloe-rgb/payload-transport-visualization)
- Public website — [Payload Transport Lab](https://arloe-rgb.github.io/payload-transport-visualization/)

### 1. Set up the visualization checkout

From the research repository root, clone the visualization repository beside it. If you already have this checkout, skip the clone and start with an up-to-date `main` branch.

```powershell
git clone https://github.com/arloe-rgb/payload-transport-visualization.git ..\payload-transport-visualization
```

### 2. Add the exported replay

Stay in the research repository with the virtual environment active. Use the `results/trial.json` export from the previous section.

```powershell
python .\visualization\publish.py ..\payload-transport-visualization add .\results\trial.json --id trial --name "Trial run" --no-sync
```

The helper validates the replay, copies it to `site/datasets/trial.json` in the visualization checkout, and adds it to `site/datasets/manifest.json`. The `--no-sync` flag limits the update to the replay and manifest.

Choose a unique ID using lowercase letters and digits, with hyphens between words if needed. The viewer displays the value of `--name`. Add `--description "Experiment notes"` to include context or `--default` to make the run load first.

### 3. Review, commit, and push from the visualization checkout

```powershell
Set-Location ..\payload-transport-visualization
git status --short
git add -- site/datasets/trial.json site/datasets/manifest.json
git diff --cached
```

If you chose a different ID, stage the matching JSON filename. Confirm that the staged changes contain only the replay and manifest, then commit and push to `main`.

```powershell
git commit -m "Publish trial run"
git push origin main
```

The visualization repository's GitHub Actions workflow deploys `site/` to GitHub Pages after the push. Once the deployment succeeds, refresh the public website and select **Trial run** from the **Recorded run** menu.
