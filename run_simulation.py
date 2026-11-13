import torch
import yaml

from models.plant_block import plant
from models.uav_block import UAV
from controllers.controller_block import PD_FF_Controller
from controllers.control_allocation_block import allocator
from trajectories.reference_trajectory_block import reference_trajectory
from visualization.simulation_logger import SimulationLogger

def run_simulation(config, *, replay_path=None):

    # Object Initialization
    payload = plant(config)
    uav = UAV(config)
    controller = PD_FF_Controller(config)
    trajectory = reference_trajectory(config)
    control_allocation = allocator(config, payload.B_mat, controller.control_cmd, trajectory.get_trajectory)
  
    # SIM Parameter Setup
    sim = config['SIM']
    dt = sim["dt"]
    duration = sim["duration"]
    num_steps = int(round(duration/dt))
    t_vec = torch.linspace(0, duration, num_steps + 1, dtype=payload.state.dtype, device=payload.state.device)

    # Init data logger
    logger = SimulationLogger(
        t_vec, payload.state, control_allocation.u, control_allocation.lamda,
        control_allocation.F_max, control_allocation.FOO_vector, trajectory.get_trajectory,
        uav=uav,
    )
    logger.log_step(0, payload.state, control_allocation.u, control_allocation.lamda)

    # Simulation Loop
    for i in range(num_steps):

        # get time
        t = t_vec[i]

        # Current allocator and payload states
        x = payload.state.clone()
        u = control_allocation.u.clone()
        x_dot = payload.run_dynamics(x, u)

        # control allocation
        u_opt = control_allocation.allocate(t, x, x_dot, dt)

        # Check control forces before applying them
        if not torch.isfinite(u_opt).all():
            raise RuntimeError(f"Invalid control forces at step {i}, t={t.item():.4f}")

        # integrate plant dynamics
        q_next = payload.integrate(u_opt, dt)

        if not torch.isfinite(q_next).all():
            raise RuntimeError(f"Invalid payload state at step {i}, t={t.item():.4f}")

        logger.log_step(i + 1, q_next, control_allocation.u, control_allocation.lamda)

    if replay_path is None:
        logger.show_plots()
    else:
        logger.export_replay(replay_path, config["payload"])

    return t_vec, logger.logged_state


def main():
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="Run the payload transport simulation.")
    parser.add_argument(
        "--export-replay", nargs="?", type=Path,
        const=Path(__file__).resolve().parent / "visualization" / "replay.json",
        metavar="PATH", help="Export completed results as JSON instead of opening Matplotlib.",
    )
    args = parser.parse_args()

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    t_vec, logged_state = run_simulation(config, replay_path=args.export_replay)
    if args.export_replay is not None:
        print(f"Replay written to {args.export_replay}")

    print("Simulation completed successfully!")
    print("Final payload state:", logged_state[-1])

if __name__ == "__main__":
    main()
