import torch

class Adaptation:

    def __init__(self, config, predictor):

        # Import config parameters
        adaptation = config["adaptation"]

        # Initial parameter estimates and projection settings
        self.w_hat = predictor.state.new_tensor(adaptation["w_init"])
        self.projection_max_norm = adaptation["projection_max_norm"]
        self.projection_tol = adaptation["projection_tol"]

        # Adaptation and prediction error gains
        self.gamma_w = self.w_hat.new_tensor(adaptation["gamma_w"])
        self.error_position_gain = adaptation["error_position_gain"]
        self.error_velocity_gain = adaptation["error_velocity_gain"]

    def project(self, w_hat, w_dot):
        # Norm bound and its gradient
        radius_sq = self.projection_max_norm**2
        tol = self.projection_tol
        f = ((1.0 + tol) * (w_hat @ w_hat) - radius_sq) / (tol * radius_sq)
        grad_f = 2.0 * (1.0 + tol) * w_hat / (tol * radius_sq)

        # Reduce the outward part of the update near the norm bound
        outward = grad_f @ w_dot
        if f > 0 and outward > 0:
            weighted_grad = self.gamma_w * grad_f
            w_dot = w_dot - f * weighted_grad * outward / (grad_f @ weighted_grad)

        return w_dot

    def run_dynamics(self, e_s, phi, foo, dfoo_dx, w_hat=None):
        if w_hat is None:
            w_hat = self.w_hat

        # Prediction error and allocator contribution
        prediction_term = (
            self.error_position_gain * e_s[:3]
            + self.error_velocity_gain * e_s[3:]
        )
        allocation_term = (dfoo_dx.T @ foo)[3:]

        # Parameter derivative
        w_dot = self.gamma_w * (phi.T @ (prediction_term + allocation_term))
        return self.project(w_hat, w_dot)
