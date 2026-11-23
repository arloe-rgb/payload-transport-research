import torch
import yaml

class PD_FF_Controller:

    def __init__(self, config, predictor=None):

        # CONTROLLER 

        # Controller Config
        controller_config = config["controller"]

        # defining gain matrices
        self.Kp = torch.diag(torch.tensor(controller_config["Kp"], dtype=torch.float64, device="cpu"))
        self.Kd = torch.diag(self.Kp.new_tensor(controller_config["Kd"]))

        # PLANT
        payload = config["payload"]

        # Physical parameters
        self.m = payload["mass"]
        self.J = payload["inertia"]
        self.g = payload["gravity"]

        # Mass matrix and gravity vector
        self.M = torch.diag(self.Kp.new_tensor([self.m, self.m, self.J]))
        self.G = self.Kp.new_tensor([0.0, -self.m*self.g, 0.0])

        # Defining error terms (may pass these into a object for calculating derivatives)
        self.prev_error = 0.0

        # State predictor for the adaptive commands
        self.predictor = predictor

    def control_cmd(self, x, x_r, xddot_r):

        # pull out position and velocities
        q = x[:3]
        qdot = x[3:]

        # pull out reference positions and velocities
        q_r = x_r[:3]
        qdot_r = x_r[3:]

        # calc error
        e = q_r - q
        edot = qdot_r - qdot

        qddot_cmd = xddot_r + self.Kd @ edot + self.Kp @ e

        tau_d = (self.M @ qddot_cmd ) - self.G

        return tau_d

    def adaptive_accel_cmd(self, x, x_hat, x_r, xddot_r):
        # Prediction error and reference tracking error
        e_s = x - x_hat
        e_r = x_r - x_hat

        # Cancel the correction applied inside the predictor
        correction = self.predictor.k_s1 * e_s[:3] + self.predictor.k_s2 * e_s[3:]

        return xddot_r + self.Kp @ e_r[:3] + self.Kd @ e_r[3:] - correction

    def adaptive_control_cmd(self, x, x_hat, x_r, xddot_r, w_hat):
        qddot_cmd = self.adaptive_accel_cmd(x, x_hat, x_r, xddot_r)

        # Estimated mass and inertia
        m_hat = self.predictor.m0 / w_hat[0]
        J_hat = self.predictor.J0 / w_hat[1]
        mass_diag = torch.stack((m_hat, m_hat, J_hat))

        # Convert the acceleration command to a wrench
        gravity_comp = qddot_cmd.new_tensor([0.0, self.g, 0.0])
        return mass_diag * (qddot_cmd + gravity_comp)
