import numpy as np
import yaml

class PD_FF_Controller:

    def __init__(self, config):

        # CONTROLLER 

        # Controller Config
        controller_config = config{"controller"}

        # defining gain matrices
        self.Kp = np.diag(controller_config["Kp"])
        self.Kd = np.diag(controller_config["Kd"])

        # PLANT
        payload = config{"payload"}

        # Physical parameters
        self.m = payload["mass"]
        self.J = payload["inertia"]
        self.g = payload["gravity"]

        # Mass matrix and gravity vector
        self.M = np.diag([self.m, self.m, self.J])
        self.G = np.array([0.0, -self.m*self.g, 0.0])

        # Defining error terms (may pass these into a object for calculating derivatives)
        self.prev_error = 0.0

    def control_cmd(self, x, x_d, xdot, xdot_d, xddot_r):
        # calc error
        e = x_d - x
        edot = xdot_d - xdot

        xddot_d = xddot_r + self.Kd @ edot + self.Kp @ e

        tau_d = (self.M @ xddot_d ) - self.G

        return tau_d
