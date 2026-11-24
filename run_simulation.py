import torch
import yaml
from time import perf_counter

from models.plant_block import plant
from models.uav_block import UAV
from models.state_predictor_block import StatePredictor
from controllers.controller_block import PD_FF_Controller
from controllers.adaptation_block import Adaptation
from controllers.control_allocation_block import allocator
from trajectories.reference_trajectory_block import reference_trajectory
from visualization.simulation_logger import SimulationLogger

def run_simulation(config, *, replay_path=None):

    started = perf_counter()

    # Object Initialization
    payload = plant(config)
    uav = UAV(config)
    trajectory = reference_trajectory(config)

    predictor = StatePredictor(config, payload.B_mat, payload.state)
    controller = PD_FF_Controller(config, predictor)
    adaptation = Adaptation(config, predictor)

    control_allocation = allocator(
        config,
        B_mat_func=payload.B_mat,
        control_cmd_func=controller.control_cmd,
        get_traj_func=trajectory.get_trajectory,
        predictor=predictor,
        adaptive_accel_cmd_func=controller.adaptive_accel_cmd,
    )
  
    # SIM Parameter Setup
    sim = config['SIM']
    dt = sim["dt"]
    duration = sim["duration"]
    num_steps = int(round(duration/dt))
    t_vec = torch.linspace(0, duration, num_steps + 1, dtype=payload.state.dtype, device=payload.state.device)

    # Collect adaptive diagnostics at each logged state
    def adaptive_snapshot(t, u, lamda, x):
        x_hat = predictor.state
        w_hat = adaptation.w_hat
        x_ref, qddot_ref = trajectory.get_trajectory(t)
        accel_cmd = controller.adaptive_accel_cmd(x, x_hat, x_ref, qddot_ref)
        uff, foo, x_hat_dot, w_hat_dot = control_allocation.adaptive_u_ff(
            t, u, lamda, x, x_hat, w_hat, adaptation
        )
        return x_hat, x_hat_dot, w_hat, w_hat_dot, accel_cmd, uff, foo

    # Initialize the logger and record the starting state
    logger = SimulationLogger(
        t_vec, payload.state, control_allocation.u, control_allocation.lamda,
        control_allocation.F_max, None, trajectory.get_trajectory,
        uav=uav, adaptive_snapshot=adaptive_snapshot,
        nominal_parameters=(predictor.m0, predictor.J0),
        true_parameters=(payload.m, payload.J),
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

        # Advance the allocator, predictor, and parameter estimates
        u_opt = control_allocation.adaptive_allocate(t, x, x_dot, dt, adaptation)

        # Check control forces before applying them
        if not torch.isfinite(u_opt).all():
            raise RuntimeError(f"Invalid control forces at step {i}, t={t.item():.4f}")

        # integrate plant dynamics
        q_next = payload.integrate(u_opt, dt)

        if not torch.isfinite(q_next).all():
            raise RuntimeError(f"Invalid payload state at step {i}, t={t.item():.4f}")

        logger.log_step(i + 1, q_next, control_allocation.u, control_allocation.lamda)

    elapsed = perf_counter() - started
    print(f"Simulation wall time: {elapsed:.2f} seconds", flush=True)

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
