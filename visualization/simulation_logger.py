"""Record and present snapshots from the simulation."""

import matplotlib.pyplot as plt
import torch

from visualization.export import write_replay


class SimulationLogger:
    def __init__(self, t_vec, state, forces, multipliers, force_limits, gradients, reference, *, uav):
        self.t_vec = t_vec.detach().clone()
        self.force_limits = force_limits.detach().clone()
        self._evaluate_gradients = gradients
        self._get_trajectory = reference
        self._uav = uav

        samples = len(self.t_vec)
        self.logged_state = state.new_zeros((samples, state.numel()))
        self.logged_forces = forces.new_zeros((samples, forces.numel()))
        self.logged_multipliers = multipliers.new_zeros((samples, multipliers.numel()))
        self.logged_gradients = forces.new_zeros((samples, forces.numel() + multipliers.numel()))
        self.logged_attachment_positions = state.new_zeros((samples, uav.N, 2))
        self.logged_uav_positions = state.new_full((samples, uav.N, 2), torch.nan)

    def log_step(self, index, state, forces, multipliers):
        """Record the current plant and allocator state at a timeline index."""
        with torch.no_grad():
            self.logged_state[index].copy_(state.detach())
            self.logged_forces[index].copy_(forces.detach())
            self.logged_multipliers[index].copy_(multipliers.detach())
            gradient = self._evaluate_gradients(
                self.t_vec[index], self.logged_forces[index], self.logged_multipliers[index],
                self.logged_state[index],
            )
            self.logged_gradients[index].copy_(gradient.detach())
            logged_state = self.logged_state[index]
            logged_forces = self.logged_forces[index]
            self.logged_attachment_positions[index].copy_(self._uav.attachment_points(logged_state))
            self.logged_uav_positions[index].copy_(self._uav.UAV_positions(logged_state, logged_forces))

    def _reference_history(self):
        with torch.no_grad():
            return torch.stack([self._get_trajectory(t)[0] for t in self.t_vec])

    def show_plots(self):
        self.plot_results()
        self.plot_forces()
        self.plot_allocator_diagnostics()
        plt.show()

    def export_replay(self, path, payload_geometry):
        write_replay(
            path, payload_geometry, self.force_limits,
            self.t_vec, self.logged_state, self._reference_history(), self.logged_forces,
            self.logged_multipliers, self.logged_gradients,
            attachment_positions=self.logged_attachment_positions,
            uav_positions=self.logged_uav_positions,
        )

    def plot_results(self):
        logged_ref = self._reference_history()

        # Convert tensors to NumPy
        t = self.t_vec.detach().cpu().numpy()
        x = self.logged_state.detach().cpu().numpy()
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


    def plot_forces(self):

        # Reshape forces into (time, UAV, x/z components)
        forces = self.logged_forces.reshape(len(self.t_vec), -1, 2)

        # Calculate force magnitude for each UAV
        force_mag = torch.linalg.vector_norm(forces, dim=2)

        # Convert to NumPy
        t = self.t_vec.detach().cpu().numpy()
        force_mag = force_mag.detach().cpu().numpy()
        F_max = self.force_limits.detach().cpu().numpy()

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


    def plot_allocator_diagnostics(self):
        time = self.t_vec.detach().cpu().numpy()
        num_forces = self.logged_forces.shape[1]
        force_labels = [
            f"UAV {i + 1} {axis}"
            for i in range(num_forces // 2)
            for axis in ("x", "z")
        ]
        wrench_labels = ["Force x", "Force z", "Pitch torque"]
        panels = (
            (self.logged_forces, force_labels, "Control inputs", "Force (N)"),
            (self.logged_multipliers, wrench_labels, "Equality-constraint multipliers", r"$\lambda$"),
            (self.logged_gradients[:, :num_forces], force_labels, "Lagrangian gradient with respect to control inputs",
             r"$\nabla_u \mathcal{L}$"),
            (self.logged_gradients[:, num_forces:], wrench_labels, "Lagrangian gradient with respect to multipliers",
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
