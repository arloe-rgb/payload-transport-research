# Payload Transport Research

Work-in-progress Python research prototype for planar payload transport using
multiple UAV attachment points. The project currently contains a payload model
draft, a proportional-derivative controller with feedforward, and placeholders
for control allocation, reference trajectories, and simulation orchestration.

## Repository layout

```text
.
|-- config.yaml                      # Payload parameters and attachment geometry
|-- controller_block.py              # PD controller with acceleration feedforward
|-- plant_block.py                   # Payload dynamics draft
|-- control_allocation_block.py      # Control allocation placeholder
|-- reference_trajectory_block.py    # Reference trajectory placeholder
|-- run_simulation.py                # Simulation entry point placeholder
|-- requirements.txt                # Python dependencies
|-- .gitignore                      # Local environments and generated outputs
`-- .gitattributes                  # Consistent text line endings
```

The modules stay at the repository root to keep this small prototype easy to
navigate and preserve the original file names.

## Environment setup

Use Python 3.10 or newer. From the repository directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dependencies are PyTorch for numerical calculations and PyYAML for reading
the configuration. Versions are not pinned yet; there is no validated runtime
environment or lockfile for this prototype.

## Configuration

`config.yaml` defines the payload mass, inertia, gravity, dimensions, and four
attachment points in the payload body frame as `[x, z]` pairs. The controller
currently uses a hard-coded gravity value of `9.81`.

## Implementation status

The project is not yet a runnable simulation. The original research modules
have been preserved during repository setup, including these unfinished items:

- `controller_block.py` implements a PD/feedforward wrench command.
- `plant_block.py` contains the initial mass, gravity, geometry, and state
  setup. Its constructor is currently named `__int__`, configuration loading
  refers to an undefined `config_path`, the PyYAML import is missing, and the
  dynamics and allocation matrix methods are incomplete. The unfinished
  assignment and empty method body currently prevent Python from parsing it.
- `control_allocation_block.py`, `reference_trajectory_block.py`, and
  `run_simulation.py` are empty placeholders.

Next steps are to complete the plant and allocation model, implement a
reference trajectory, connect the blocks in `run_simulation.py`, and validate
the resulting dynamics and controller. No automated test suite is present yet.
