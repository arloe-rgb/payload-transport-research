"""Record and present snapshots from the simulation."""

import matplotlib.pyplot as plt
import torch

from visualization.export import write_replay


class SimulationLogger:
    def __init__(
        self, t_vec, state, forces, multipliers, force_limits, gradients, reference,
        *, uav, adaptive_snapshot=None, nominal_parameters=None, true_parameters=None,
    ):
        """The adaptive snapshot returns x_hat, x_hat_dot, w_hat, w_hat_dot,
        commanded acceleration, feedforward, and optimality gradients.
        """
        self.t_vec = t_vec.detach().clone()
        self.force_limits = force_limits.detach().clone()
        self._evaluate_gradients = gradients
        self._get_trajectory = reference
        self._uav = uav
        self._adaptive_snapshot = adaptive_snapshot

        samples = len(self.t_vec)
        self.logged_state = state.new_zeros((samples, state.numel()))
        self.logged_forces = forces.new_zeros((samples, forces.numel()))
        self.logged_multipliers = multipliers.new_zeros((samples, multipliers.numel()))
        self.logged_gradients = forces.new_zeros((samples, forces.numel() + multipliers.numel()))
        self.logged_attachment_positions = state.new_zeros((samples, uav.N, 2))
        self.logged_uav_positions = state.new_full((samples, uav.N, 2), torch.nan)

        if adaptive_snapshot is not None:
            self._nominal_parameters = (
                state.new_tensor(nominal_parameters) if nominal_parameters is not None else None
            )
            self._true_parameters = true_parameters
            self.logged_predicted_state = torch.zeros_like(self.logged_state)
            self.logged_predicted_state_dot = torch.zeros_like(self.logged_state)
            self.logged_w_hat = state.new_zeros((samples, 2))
            self.logged_w_hat_dot = torch.zeros_like(self.logged_w_hat)
            self.logged_accel_command = torch.zeros_like(self.logged_multipliers)
            self.logged_estimated_accel = torch.zeros_like(self.logged_multipliers)
            self.logged_feedforward = torch.zeros_like(self.logged_gradients)

    def log_step(self, index, state, forces, multipliers):
        """Record the current plant and allocator state at a timeline index."""
        with torch.no_grad():
            self.logged_state[index].copy_(state.detach())
            self.logged_forces[index].copy_(forces.detach())
            self.logged_multipliers[index].copy_(multipliers.detach())
            logged_state = self.logged_state[index]
            logged_forces = self.logged_forces[index]
            sample = (self.t_vec[index], logged_forces, self.logged_multipliers[index], logged_state)
            if self._adaptive_snapshot is None:
                gradient = self._evaluate_gradients(*sample)
            else:
                x_hat, x_hat_dot, w_hat, w_hat_dot, accel_cmd, uff, gradient = self._adaptive_snapshot(*sample)
                self.logged_predicted_state[index].copy_(x_hat)
                self.logged_predicted_state_dot[index].copy_(x_hat_dot)
                self.logged_w_hat[index].copy_(w_hat)
                self.logged_w_hat_dot[index].copy_(w_hat_dot)
                self.logged_accel_command[index].copy_(accel_cmd)
                # The multiplier gradient is commanded minus estimated acceleration.
                accel_residual = gradient[logged_forces.numel():]
                self.logged_estimated_accel[index].copy_(accel_cmd - accel_residual)
                self.logged_feedforward[index].copy_(uff)
            self.logged_gradients[index].copy_(gradient.detach())
            self.logged_attachment_positions[index].copy_(self._uav.attachment_points(logged_state))
            self.logged_uav_positions[index].copy_(self._uav.UAV_positions(logged_state, logged_forces))

    def _reference_history(self):
        with torch.no_grad():
            return torch.stack([self._get_trajectory(t)[0] for t in self.t_vec])

    def show_plots(self):
        self.plot_results()
        self.plot_forces()
        self.plot_allocator_diagnostics()
        if self._adaptive_snapshot is not None:
            self.plot_prediction_error()
            self.plot_parameter_estimates()
            self.plot_acceleration()
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

        adaptive = self._adaptive_snapshot is not None
        x_hat = self.logged_predicted_state.cpu().numpy() if adaptive else None
        columns = 2 if adaptive else 1
        fig, axs = plt.subplots(3, columns, figsize=(5 * columns + 5, 8), sharex=True, squeeze=False)
        labels = (
            "X Position (m)", "Z Position (m)", "Pitch (rad)",
            "X Velocity (m/s)", "Z Velocity (m/s)", "Pitch Rate (rad/s)",
        )

        for i, ax in enumerate(axs.T.flat):
            ax.plot(t, x[:, i], label="Actual", linewidth=2)
            ax.plot(t, x_ref[:, i], "--", label="Reference", linewidth=2)
            if adaptive:
                ax.plot(t, x_hat[:, i], ":", label="Predicted", linewidth=2)
            ax.set_ylabel(labels[i])
            ax.grid(True)
            ax.legend()

        for ax in axs[-1]:
            ax.set_xlabel("Time (s)")

        fig.suptitle("Payload Trajectory Tracking")
        fig.tight_layout()

        plt.figure(figsize=(8, 6))

        plt.plot(x[:, 0], x[:, 1], label="Actual Path")
        plt.plot(x_ref[:, 0], x_ref[:, 1], "--", label="Reference Path")
        if adaptive:
            plt.plot(x_hat[:, 0], x_hat[:, 1], ":", label="Predicted Path")

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
        constraint_labels = ["Force x", "Force z", "Pitch torque"]
        residual_labels = constraint_labels
        residual_title = "Lagrangian gradient with respect to multipliers"
        if self._adaptive_snapshot is not None:
            constraint_labels = ["X acceleration", "Z acceleration", "Pitch angular acceleration"]
            residual_labels = ["X (m/s²)", "Z (m/s²)", "Pitch (rad/s²)"]
            residual_title = "Acceleration constraint residuals"
        panels = (
            (self.logged_forces, force_labels, "Control inputs", "Force (N)"),
            (self.logged_multipliers, constraint_labels, "Equality-constraint multipliers", r"$\lambda$"),
            (self.logged_gradients[:, :num_forces], force_labels, "Lagrangian gradient with respect to control inputs",
             r"$\nabla_u \mathcal{L}$"),
            (self.logged_gradients[:, num_forces:], residual_labels, residual_title,
             r"$\nabla_\lambda \mathcal{L}$"),
        )
        if self._adaptive_snapshot is not None:
            panels += (
                (self.logged_feedforward[:, :num_forces], force_labels, "Adaptive feedforward for force gradients",
                 r"$u_{\mathrm{ff}}$"),
                (self.logged_feedforward[:, num_forces:], constraint_labels, "Adaptive feedforward for constraint residuals",
                 r"$u_{\mathrm{ff}}$"),
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

    def plot_prediction_error(self):
        time = self.t_vec.cpu().numpy()
        error = (self.logged_state - self.logged_predicted_state).cpu().numpy()
        labels = (
            "X position (m)", "Z position (m)", "Pitch (rad)",
            "X velocity (m/s)", "Z velocity (m/s)", "Pitch rate (rad/s)",
        )
        fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
        for i, ax in enumerate(axs.T.flat):
            ax.plot(time, error[:, i])
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)
        for ax in axs[-1]:
            ax.set_xlabel("Time (s)")
        fig.suptitle("Prediction error (measured minus predicted)")

    def plot_parameter_estimates(self):
        time = self.t_vec.cpu().numpy()
        ratios = self.logged_w_hat.cpu().numpy()
        rates = self.logged_w_hat_dot.cpu().numpy()
        rows = 3 if self._nominal_parameters is not None else 2
        fig, axs = plt.subplots(rows, 2, figsize=(12, 2.5 * rows), sharex=True, layout="constrained")
        axs[0, 0].set_title("Mass")
        axs[0, 1].set_title("Pitch inertia")
        if self._nominal_parameters is not None:
            nonzero_ratios = self.logged_w_hat.masked_fill(self.logged_w_hat == 0, torch.nan)
            physical_estimates = (self._nominal_parameters / nonzero_ratios).cpu().numpy()
            axs[2, 0].set_ylabel("Estimated mass (kg)")
            axs[2, 1].set_ylabel("Estimated inertia (kg·m²)")
        for i in range(2):
            axs[0, i].plot(time, ratios[:, i])
            axs[0, i].axhline(1, color="black", linestyle="--", linewidth=1)
            axs[0, i].set_ylabel("Estimated ratio")
            axs[1, i].plot(time, rates[:, i])
            axs[1, i].set_ylabel("Projected ratio rate (1/s)")
            if self._nominal_parameters is not None:
                ax = axs[2, i]
                ax.plot(time, physical_estimates[:, i], label="Estimate")
                if self._true_parameters is not None:
                    ax.axhline(self._true_parameters[i], color="black", linestyle="--", label="True value")
                    ax.legend()
            axs[-1, i].set_xlabel("Time (s)")
        for ax in axs.flat:
            ax.grid(True, alpha=0.3)
        fig.suptitle("Adaptive parameter estimates")

    def plot_acceleration(self):
        time = self.t_vec.cpu().numpy()
        command = self.logged_accel_command.cpu().numpy()
        model = self.logged_estimated_accel.cpu().numpy()
        prediction = self.logged_predicted_state_dot[:, 3:].cpu().numpy()
        labels = ("X (m/s²)", "Z (m/s²)", "Pitch (rad/s²)")
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
        for i, ax in enumerate(axs):
            ax.plot(time, command[:, i], label="Commanded")
            ax.plot(time, model[:, i], "--", label="Estimated model")
            ax.plot(time, prediction[:, i], ":", label="Predictor derivative")
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)
            ax.legend()
        axs[-1].set_xlabel("Time (s)")
        fig.suptitle("Adaptive acceleration diagnostics")
