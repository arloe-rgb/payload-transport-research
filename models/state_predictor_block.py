import torch

class StatePredictor:

    def __init__(self, config, B_mat_func, initial_state):

        # Import config parameters
        predictor = config["state_predictor"]
        payload = config["payload"]

        # Allocation matrix function from the plant
        self.B_mat = B_mat_func

        # Initial predicted state [x, z, theta, x_dot, z_dot, theta_dot]
        self.state = initial_state.clone()

        # Nominal mass and inertia
        self.m0 = predictor["nominal_mass"]
        self.J0 = predictor["nominal_inertia"]

        # Positive predictor gains
        self.k_s1 = predictor["k_s1"]
        self.k_s2 = predictor["k_s2"]

        # Gravity acceleration and predictor matrices
        self.tau0 = self.state.new_tensor([0.0, -payload["gravity"], 0.0])
        identity = torch.eye(3, dtype=self.state.dtype, device=self.state.device)
        self.A_x = self.state.new_zeros((6, 6))
        self.A_x[:3, 3:] = identity
        self.B_x = self.state.new_zeros((6, 3))
        self.B_x[3:] = identity
        self.A_s = self.A_x.clone()
        self.A_s[3:, :3] = -self.k_s1 * identity
        self.A_s[3:, 3:] = -self.k_s2 * identity

    def regressor(self, x, u):
        # Compute the wrench at the measured payload angle
        B = self.B_mat(x[2])
        wrench = B @ u

        # Build the mass and inertia columns
        phi = wrench.new_zeros((3, 2))
        phi[:2, 0] = wrench[:2] / self.m0
        phi[2, 1] = wrench[2] / self.J0

        return phi

    def run_dynamics(self, x, u, w_hat, x_hat=None):
        if x_hat is None:
            x_hat = self.state

        # Prediction error
        e_s = x - x_hat

        # Estimated payload acceleration
        phi = self.regressor(x, u)
        tau_hat = self.tau0 + phi @ w_hat

        # Predicted state derivative
        x_hat_dot = self.A_x @ x + self.B_x @ tau_hat - self.A_s @ e_s
        
        return x_hat_dot
