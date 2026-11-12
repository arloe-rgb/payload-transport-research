import torch
import yaml

class PD_FF_Controller:

    def __init__(self, config):

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
