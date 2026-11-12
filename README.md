# Payload Transport Research

Work-in-progress Python research prototype for planar payload transport using
multiple UAV attachment points. The project currently contains a payload
model, a proportional-derivative controller with feedforward, a control
allocator, and reference trajectories. The simulation plots payload tracking
and the forces allocated to each UAV.

## Repository layout

```text
.
|-- config.yaml                      # Payload parameters and attachment geometry
|-- controllers/
|   |-- controller_block.py          # PD controller with acceleration feedforward
|   `-- control_allocation_block.py  # UAV force allocation
|-- models/plant_block.py            # Payload dynamics
|-- trajectories/reference_trajectory_block.py
|-- run_simulation.py                # Simulation entry point and plotting
|-- requirements.txt                # Python dependencies
|-- .gitignore                      # Local environments and generated outputs
`-- .gitattributes                  # Consistent text line endings
```

## Python Environment Setup

On a new computer, clone the repository and open it in VS Code:

```powershell
git clone https://github.com/arloe-rgb/payload-transport-research.git
Set-Location .\payload-transport-research
code .
```

If you already have a checkout, open a PowerShell terminal in its root, run
`git pull`, and open that folder in VS Code. Use **File > Open Folder** if the
`code` command is unavailable. Install the Microsoft Python extension in VS Code.

With a local Python 3.10 or newer installed, run these commands from the
repository root in the VS Code PowerShell terminal. The setup was verified
with Python 3.14.3. Run the first command only if you need to create `.venv`.
If the `py` launcher is unavailable, check `python --version` and use
`python -m venv .venv` with your local Python installation.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
python -c "import torch, yaml, numpy, matplotlib; print(torch.__version__)"
```

The printed interpreter should end in `.venv\Scripts\python.exe`. The workspace
setting points the Python extension to that interpreter. If VS Code previously
saved a different interpreter for the folder, select the project interpreter
explicitly as described under Troubleshooting.

PyTorch handles numerical calculations, PyYAML reads the configuration, and
Matplotlib plots the results. The plotting code converts tensors to NumPy
arrays, so NumPy is listed explicitly. Requirements are unpinned, so the
resolved versions can vary with Python version and install date.

## Running the Simulation

From the repository root, which contains both `run_simulation.py` and
`config.yaml`, run:

```powershell
.\.venv\Scripts\Activate.ps1
python .\run_simulation.py
deactivate
```

Close the plot windows to let the script finish before deactivating the
environment. You can also run the script directly with
`.\.venv\Scripts\python.exe .\run_simulation.py`.

## Browser Replay

From the repository root, use the existing Python environment to export a
completed simulation and serve the viewer locally:

```powershell
.\.venv\Scripts\python.exe .\run_simulation.py --export-replay
.\.venv\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1 --directory visualization
```

Open [Payload Transport Lab](http://127.0.0.1:8000/). The viewer loads
`visualization/replay.json` automatically. Use **Open replay** to load a
different export. You can also open `visualization/index.html` directly and
select the JSON manually without starting a server. To save multiple runs,
give the export a path, for example `--export-replay .\results\trial.json`.

Use **Play**, the timeline, or the step buttons to review a run. The
collapsible research diagnostics show tracking, force magnitudes and limits,
force components, multipliers, and both Lagrangian gradient norms. Click
any diagnostic chart to move the shared playhead.

In the 3D scene, drag to orbit, scroll to zoom, and right-drag to pan. Choose
**Front (X–Z)** for the original planar view, **Side (Y–Z)** or **Top (X–Y)**
to inspect depth, or **Perspective** for an angled view. **Reset camera**
restores the selected preset. **Follow payload** moves the camera with the
payload. The scene uses the local Three.js bundle and requires WebGL 2.

The export contains actual PyTorch states, reference states, and allocator
histories at each simulation timestamp. The browser interpolates between
samples for smooth playback. UAV positions are illustrative and use the
logged force directions and the cable length set in the viewer. Cable
length, arrow scale, and payload depth only affect the display. The payload
center, attachment points, force arrows, and trajectories stay on Y = 0.

Refresh the page after regenerating the default replay. Use Ctrl+C to stop
the local server. Running `run_simulation.py` without the export flag still
opens the existing Matplotlib figures.

## Working Across Multiple Computers

Each computer creates and maintains its own `.venv`. GitHub synchronizes the
tracked project files, including source code, configuration, documentation, and
`requirements.txt`. Both `.venv` and `.venv-1` are ignored. Create the environment
locally on each computer because a virtual environment records paths to its
base Python installation.

After pulling changes, activate `.venv` and run
`python -m pip install -r requirements.txt` to pick up dependency changes.

## Troubleshooting

- **PowerShell blocks activation.** Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`
  in the current terminal and retry activation. If a managed policy still blocks
  it, use `.\.venv\Scripts\python.exe` directly for installation and execution.
- **VS Code uses a different interpreter.** Open **Python: Select Interpreter**
  from the Command Palette, choose **Enter interpreter path**, and select
  `.venv\Scripts\python.exe` from this checkout. Open a fresh terminal and check
  `python -c "import sys; print(sys.executable)"`.
- **A package is missing.** From the repository root, run
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and
  `.\.venv\Scripts\python.exe -m pip check`.
- **The environment is broken.** Deactivate it and close terminals and debug
  sessions using it. Move `.venv` to a backup location outside the checkout,
  then repeat the setup with the local Python installation. Verify the imports
  and interpreter before removing the backup.

## Configuration

`config.yaml` defines the payload mass, inertia, gravity, dimensions, and four
attachment points in the payload body frame as `[x, z]` pairs. It also defines
the controller gains, allocation constraints, simulation timing, and reference
trajectory.

## Implementation status

The entry point loads the configuration, runs the reference tracking
simulation, and plots payload position, orientation, path, and UAV forces.
No automated test suite is present yet.
