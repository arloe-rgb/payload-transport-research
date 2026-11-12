# Payload Transport Lab

[Open the interactive viewer](https://arloe-rgb.github.io/payload-transport-visualization/).

This static site replays completed runs of a planar collaborative payload
transport simulation in 3D. UAV positions and cable lengths are illustrative.
The browser reads recorded states, reference poses, forces and optimization
diagnostics. Visitors need a WebGL 2 browser.

## Recorded runs

`site/datasets/manifest.json` is a version 1 manifest. Each entry has an ID,
name, duration in seconds, relative data file, run metadata, and optional
description. The `default` ID determines the run loaded at startup.

Each version 1 replay stores payload width and height (m), attachment points
in body-frame `[x,z]` pairs (m), per-UAV force limits (N), and these histories
with one row per timestamp:

| Field | Values |
| --- | --- |
| `time` | Simulation timestamps in seconds |
| `state`, `reference` | `[x,z,theta,vx,vz,omega]` in SI units and radians |
| `u` | `[Fx1,Fz1,Fx2,Fz2,...]` in N |
| `lambda` | Optional equality multipliers in force-x, force-z, torque order |
| `gradientU`, `gradientLambda` | Optional Lagrangian gradients in the corresponding variable order |

## Deployment

The GitHub Actions workflow uploads `site/` and deploys it to GitHub Pages
on pushes to `main`. The repository's Pages source is GitHub Actions. The
artifact contains static files and requires no build or server-side runtime.

The viewer is maintained in the research checkout. To add a run, export it
there with `run_simulation.py --export-replay PATH` and use
`visualization/publish.py PUBLIC_CHECKOUT add PATH --id ID --name NAME`.
Review, commit, and push from the public checkout. Use the same helper's
`sync` command to update only the viewer and deployment files.
