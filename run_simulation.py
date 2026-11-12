import torch
import yaml
import matplotlib.pyplot as plt

from models.plant_block import plant
from controllers.controller_block import PD_FF_Controller
from controllers.control_allocation_block import allocator
from trajectories.reference_trajectory_block import reference_trajectory

def run_simulation(config, *, replay_path=None):

    # Object Initialization
    payload = plant(config)
    controller = PD_FF_Controller(config)
    trajectory = reference_trajectory(config)
    control_allocation = allocator(config, payload.B_mat, controller.control_cmd, trajectory.get_trajectory)
  
    # SIM Parameter Setup
    sim = config['SIM']
    dt = sim["dt"]
    duration = sim["duration"]
    num_steps = int(round(duration/dt))
    t_vec = torch.linspace(0, duration, num_steps + 1, dtype=payload.state.dtype, device=payload.state.device)

    # DATA logging setup
    logged_state = payload.state.new_zeros((num_steps + 1, payload.state.numel()))
    logged_state[0] = payload.state.clone()
    num_forces = control_allocation.u.numel()
    num_multipliers = control_allocation.lamda.numel()
    logged_forces = control_allocation.u.new_zeros((num_steps + 1, num_forces))
    logged_multipliers = control_allocation.lamda.new_zeros((num_steps + 1, num_multipliers))
    logged_gradients = control_allocation.u.new_zeros((num_steps + 1, num_forces + num_multipliers))

    def log_allocator_sample(index, state):
        # FOO_vector computes B and the desired wrench from this sample's state and time.
        with torch.no_grad():
            logged_forces[index].copy_(control_allocation.u.detach())
            logged_multipliers[index].copy_(control_allocation.lamda.detach())
            gradient = control_allocation.FOO_vector(
                t_vec[index], logged_forces[index], logged_multipliers[index], state.detach()
            )
            logged_gradients[index].copy_(gradient.detach())

    log_allocator_sample(0, payload.state)

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

        # Store states
        logged_state[i + 1] = q_next
        log_allocator_sample(i + 1, q_next)

    if replay_path is None:
        plot_results(t_vec, logged_state, trajectory)
        plot_forces(t_vec, logged_forces, control_allocation.F_max)
        plot_allocator_diagnostics(t_vec, logged_forces, logged_multipliers, logged_gradients)
        plt.show()
    else:
        from visualization.export import write_replay

        logged_reference = torch.stack([trajectory.get_trajectory(t)[0] for t in t_vec])
        write_replay(
            replay_path, config["payload"], control_allocation.F_max,
            t_vec, logged_state, logged_reference, logged_forces,
            logged_multipliers, logged_gradients,
        )

    return t_vec, logged_state

def plot_results(t_vec, logged_state, trajectory):

    # Calculate reference trajectory at each time step
    logged_ref = torch.stack([
        trajectory.get_trajectory(t)[0]
        for t in t_vec
    ])

    # Convert tensors to NumPy
    t = t_vec.detach().cpu().numpy()
    x = logged_state.detach().cpu().numpy()
    x_ref = logged_ref.detach().cpu().numpy()

    # Plot position and orientation
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ["X Position (m)", "Z Position (m)", "Theta (rad)"]

    for i in range(3):

        axs[i].plot(t, x[:, i], label="Actual", linewidth=2)
        axs[i].plot(t, x_ref[:, i], "--", label="Reference", linewidth=2)

        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Time (s)")

    plt.suptitle("Payload Trajectory Tracking")
    plt.tight_layout()

    plt.figure(figsize=(8, 6))

    plt.plot(x[:, 0], x[:, 1], label="Actual Path")
    plt.plot(x_ref[:, 0], x_ref[:, 1], "--", label="Reference Path")

    plt.xlabel("X Position (m)")
    plt.ylabel("Z Position (m)")
    plt.title("Payload Path Tracking")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()


def plot_forces(t_vec, logged_forces, F_max):

    # Reshape forces into (time, UAV, x/z components)
    forces = logged_forces.reshape(len(t_vec), -1, 2)

    # Calculate force magnitude for each UAV
    force_mag = torch.linalg.vector_norm(forces, dim=2)

    # Convert to NumPy
    t = t_vec.detach().cpu().numpy()
    force_mag = force_mag.detach().cpu().numpy()
    F_max = F_max.detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 6))

    for i in range(force_mag.shape[1]):

        plt.plot(
            t, force_mag[:, i],
            label=f"UAV {i+1} Force"
        )

        plt.axhline(
            F_max[i],
            linestyle="--",
            label=f"UAV {i+1} Max Force"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Force Magnitude (N)")
    plt.title("UAV Force Magnitudes vs Maximum Force")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_allocator_diagnostics(t_vec, logged_forces, logged_multipliers, logged_gradients):
    time = t_vec.detach().cpu().numpy()
    num_forces = logged_forces.shape[1]
    force_labels = [
        f"UAV {i + 1} {axis}"
        for i in range(num_forces // 2)
        for axis in ("x", "z")
    ]
    wrench_labels = ["Force x", "Force z", "Pitch torque"]
    panels = (
        (logged_forces, force_labels, "Control inputs", "Force (N)"),
        (logged_multipliers, wrench_labels, "Equality-constraint multipliers", r"$\lambda$"),
        (logged_gradients[:, :num_forces], force_labels, "Lagrangian gradient with respect to control inputs",
         r"$\nabla_u \mathcal{L}$"),
        (logged_gradients[:, num_forces:], wrench_labels, "Lagrangian gradient with respect to multipliers",
         r"$\nabla_\lambda \mathcal{L}$"),
    )
    for index, (values, labels, title, ylabel) in enumerate(panels):
        _, ax = plt.subplots(num=title, figsize=(11, 6), layout="constrained")
        values = values.detach().cpu().numpy()
        for column, label in enumerate(labels):
            ax.plot(time, values[:, column], label=label)
        if index >= 2:
            ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Zero")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4, fontsize="small")


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
