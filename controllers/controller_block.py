import numpy as np
import yaml

class PD_FF_Controller:

    def __init__(self, config):

        # CONTROLLER 

        # Controller Config
        controller_config = config["controller"]

        # defining gain matrices
        self.Kp = np.diag(controller_config["Kp"])
        self.Kd = np.diag(controller_config["Kd"])

        # PLANT
        payload = config["payload"]

        # Physical parameters
        self.m = payload["mass"]
        self.J = payload["inertia"]
        self.g = payload["gravity"]

        # Mass matrix and gravity vector
        self.M = np.diag([self.m, self.m, self.J])
        self.G = np.array([0.0, -self.m*self.g, 0.0])

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
