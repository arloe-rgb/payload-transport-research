import torch
import yaml
import matplotlib.pyplot as plt

from models.plant_block import plant
from controllers.controller_block import PD_FF_Controller
from controllers.control_allocation_block import allocator
from trajectories.reference_trajectory_block import reference_trajectory

def run_simulation(config):

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

    plot_results(t_vec, logged_state, trajectory)

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

    plt.show()

def main():

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    t_vec, logged_state = run_simulation(config)

    print("Simulation completed successfully!")
    print("Final payload state:", logged_state[-1])

if __name__ == "__main__":
    main()
